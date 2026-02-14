# Cosmos Reason 2 interface: structured perception and strategic explanation.
# Layer 1: schema-constrained hazard extraction.
# Layer 3: explanation of deterministic outcomes.

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from configs.config import get_config
from .hazard_schema import HAZARD_TYPES

_model = None
_processor = None

# Evaluation counters (log for judge credibility)
_json_parse_failures = 0
_normalization_parse_failures = 0


def get_cosmos_eval_counts() -> dict:
    return {
        "json_parse_failures": _json_parse_failures,
        "normalization_parse_failures": _normalization_parse_failures,
    }


# JSON schema for NIM structured output. Enum values must match HAZARD_TYPES.
HAZARD_SCHEMA = {
    "type": "object",
    "properties": {
        "hazards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(HAZARD_TYPES.keys()),
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical", "contextual"],
                    },
                },
                "required": ["type", "severity"],
                "additionalProperties": False,
            },
        },
        "visibility_status": {
            "type": "string",
            "enum": ["clear", "occluded", "unknown"],
        },
    },
    "required": ["hazards", "visibility_status"],
    "additionalProperties": False,
}

SYS_EXTRACT = {
    "role": "system",
    "content": [{"type": "text", "text": "You are a helpful assistant specialized in identifying physical hazards and visibility conditions from media (images or video clips)."}],
}

SYS_NORMALIZE = {
    "role": "system",
    "content": [{"type": "text", "text": "You are a helpful assistant specialized in mapping hazard descriptions into a fixed canonical hazard taxonomy."}],
}

SYS_EXPLAIN = {
    "role": "system",
    "content": [{"type": "text", "text": "You are a helpful assistant specialized in explaining safety decisions using clear physical reasoning and the required format."}],
}


def _preprocess_json(text: str) -> str:
    """Strip markdown fences and extract JSON object."""
    text = text.strip()
    if "```" in text:
        # Remove ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


def _try_repair_normalize_json(json_str: str) -> str:
    """Attempt to fix common model JSON issues before parsing (Layer 2 output)."""
    # Fix missing comma before "visibility_status" (model often omits it)
    json_str = re.sub(r'"([^"]+)"\s+"visibility_status"', r'"\1", "visibility_status"', json_str)
    # Remove visibility_status from inside hazard objects (schema expects only at top level)
    json_str = re.sub(r',\s*"visibility_status"\s*:\s*"[^"]*"', "", json_str)
    json_str = re.sub(r'"visibility_status"\s*:\s*"[^"]*"\s*,?\s*', "", json_str)
    # Fix trailing commas before ] or }
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    return json_str


def _load_model():
    global _model, _processor
    if _model is None:
        cfg = get_config().cosmos
        model_name = cfg.model_name
        offline = cfg.offline
        load_kw = {"local_files_only": offline} if offline else {}
        _processor = AutoProcessor.from_pretrained(model_name, **load_kw)
        for attn_impl in cfg.attention_preference:
            try:
                _model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    attn_implementation=attn_impl,
                    **load_kw,
                )
                break
            except Exception:
                continue
        if _model is None:
            raise RuntimeError(f"Failed to load model with any of {cfg.attention_preference}")
        if torch.cuda.is_available():
            dev = next(_model.parameters()).device
            if "cpu" in str(dev):
                import warnings
                warnings.warn(f"Model loaded on {dev}; expected cuda. Check device_map.")
        if cfg.timing:
            attn = getattr(_model.config, "_attn_implementation", None) or getattr(_model.config, "attn_implementation", "?")
            print("Attention implementation:", attn)
    return _model, _processor


def _build_media_content(media_path: str, media_type: str, fps: int = 4) -> dict:
    if media_type == "image":
        img = Image.open(media_path).convert("RGB")
        cfg = get_config().cosmos.image
        if cfg.resize:
            img = img.resize(cfg.resize_hw)
        return {"type": "image", "image": img}
    if media_type == "video":
        abs_path = str(Path(media_path).resolve())
        uri = Path(abs_path).as_uri()
        return {"type": "video", "video": uri, "fps": fps}
    raise ValueError(f"Unknown media_type: {media_type!r}, must be 'image' or 'video'")


# Layer 1: Structured hazard extraction from image or video. Returns schema-constrained hazards + visibility_status.
# Cosmos does NOT output safety or recommendations. Cosmos only extracts hazards from the media.
def query_cosmos_extract(media_path: str, media_type: str = "image", fps: int = 4) -> dict:
    model, processor = _load_model()

    media_item = _build_media_content(media_path, media_type, fps)

    prompt = """You are a structured hazard extraction engine for physical safety reasoning.
You do NOT perform navigation, planning, or recommendations.
You ONLY extract hazards and visibility status.

Analyze the provided media (image or video clip) and return ONLY valid JSON.

Describe each hazard in clear, specific terms based on what you observe.
Examples: "incomplete stairs", "debris pile", "exposed electrical wiring", "collapsed staircase", "slippery floor".
For severity use: low, medium, high, critical, or contextual.

Field descriptions:
- hazards: list of hazard objects.
- type: describe the hazard in clear, specific terms (e.g. incomplete stairs, debris pile, exposed wiring).
- severity: one of low, medium, high, critical, contextual.
- visibility_status: describes forward path visibility.

Format:
{
  "hazards": [
    {"type": "<hazard_description>", "severity": "<low|medium|high|critical|contextual>"}
  ],
  "visibility_status": "<clear|occluded|unknown>"
}

If no hazards are visible, return: {"hazards":[],"visibility_status":"clear"}
If uncertain, use "unknown" for visibility_status.
Never return null. Never omit required fields.

Rules:
- Use descriptive hazard names based on what you see (e.g. incomplete stairs, debris pile, exposed wiring).
- You may include multiple hazards in the array.
- Output valid JSON only. No commentary.
- Use compact JSON: no extra newlines, indentation, or whitespace.

Stop generation immediately after the closing brace.
"""

    messages = [
        SYS_EXTRACT,
        {"role": "user", "content": [media_item, {"type": "text", "text": prompt}]},
    ]

    t0 = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )
    inputs = inputs.to(model.device)
    t1 = time.time()

    cfg = get_config().cosmos
    max_new_tokens = cfg.video.max_new_tokens if media_type == "video" else cfg.image.max_new_tokens

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=cfg.generation.do_sample,
            use_cache=cfg.generation.use_cache,
        )
    t2 = time.time()

    if cfg.timing:
        print("Input prep:", round(t1 - t0, 2))
        print("Generate:", round(t2 - t1, 2))
        text_cfg = getattr(model.config, "text_config", None)
        use_cache_val = getattr(text_cfg, "use_cache", None) if text_cfg else getattr(model.config, "use_cache", None)
        print("model.config.use_cache:", use_cache_val)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    if cfg.timing:
        print("Layer 1 raw output:", repr(decoded[:500] + "..." if len(decoded) > 500 else decoded))
    # Return raw output; Layer 2 (normalize) interprets and structures it.
    return {"raw": decoded}


def query_cosmos_structured(image_path: str) -> dict:
    """Deprecated: use query_cosmos_extract(image_path, media_type='image') instead."""
    return query_cosmos_extract(image_path, media_type="image")


# Layer 2: Taxonomy normalization. Cosmos receives raw Layer 1 output and extracts/maps to canonical types.
def query_cosmos_normalize(raw_extraction: str) -> dict:
    """Interpret raw Layer 1 output and map hazards to canonical types. Text-only, no image."""
    model, processor = _load_model()

    allowed_types = ", ".join(HAZARD_TYPES.keys())

    prompt = f"""You are a hazard taxonomy normalizer. The following is raw output from a hazard extraction model that analyzed an image or video. Extract all hazards mentioned and map each to a canonical type.

Raw Layer 1 output:
{raw_extraction}

Allowed canonical types: {allowed_types}

Examples: incomplete stairs, collapsed staircase, missing steps, hole, excavation → hole; debris pile, construction debris → unstable_debris_stack; exposed wiring, electrical hazard → electrical_exposure; blocked path, restricted exit → restricted_escape_route; partial floor collapse, floor collapse → partial_floor_collapse; danger sign → map based on what it warns about.

Return ONLY valid JSON:
{{"hazards":[{{"type":"<canonical>","severity":"<low|medium|high|critical|contextual>"}}],"visibility_status":"<clear|occluded|unknown>"}}

Rules:
- Extract every hazard from the raw output. Map each to exactly one canonical type.
- Preserve the original severity (low, medium, high, critical, contextual).
- Never invent hazards. Only map the ones given.
- Extract visibility_status from the raw output if present; otherwise use "unknown".
- Output compact JSON only. No commentary."""

    messages = [
        SYS_NORMALIZE,
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    cfg = get_config().cosmos
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens_normalize,
            do_sample=cfg.generation.do_sample,
            use_cache=cfg.generation.use_cache,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    json_str = _preprocess_json(decoded)
    parsed = None
    first_error = None
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        first_error = e
        json_str = _try_repair_normalize_json(json_str)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            pass
    if parsed is None:
        global _normalization_parse_failures
        _normalization_parse_failures += 1
        if cfg.timing:
            print("Layer 2 JSON parse failed:", first_error or "parse failed")
            print("Layer 2 decoded snippet:", repr(decoded[:400] + "..." if len(decoded) > 400 else decoded))
        return {"hazards": [], "visibility_status": "unknown"}

    raw_hazards = parsed.get("hazards", [])
    filtered = [
        {"type": h["type"], "severity": h.get("severity", "medium")}
        for h in raw_hazards
        if isinstance(h, dict) and h.get("type") in HAZARD_TYPES
    ]
    if cfg.timing:
        print("Layer 2 hazards (before filter):", raw_hazards)
        dropped = [h.get("type") for h in raw_hazards if isinstance(h, dict) and h.get("type") not in HAZARD_TYPES]
        if dropped:
            print("Layer 2 hazard types dropped (not in HAZARD_TYPES):", dropped)
        print("Layer 2 hazards (after filter):", filtered)
    parsed["hazards"] = filtered
    vs = parsed.get("visibility_status", "unknown")
    parsed["visibility_status"] = vs if vs in ("clear", "occluded", "unknown") else "unknown"
    return parsed


# Layer 3: Strategic explanation. Cosmos receives hazards, safety, recommendation and produces clear explanation, human instructions, tactical justification. Cosmos explains the deterministic outcome; it does not compute it.
def query_cosmos_explanation(context: dict) -> str:
    model, processor = _load_model()

    prompt = f"""Explain the deterministic safety decision below.

Input:
Hazards: {context.get("hazards", [])}
Drone safety: {context.get("safety", {}).get("drone_path_safety")}
Human safety: {context.get("safety", {}).get("human_follow_safety")}
Recommendation: {context.get("recommendation", {}).get("recommendation")}
Fallback available: {context.get("fallback_available", False)}

Agent constraints:
Drone:
- Can fly over gaps
- Does not require stable ground
- Sensitive to electricity, heat, entanglement

Human:
- Requires stable ground
- Cannot fly over gaps
- Sensitive to instability, collapse, confined air

Return EXACTLY this structure:

Hazards:
<text>

Drone Safety:
<text>

Human Safety:
<text>

Reasoning:
<text>

Rules:
- Use physically grounded language. Refer to actual constraints (e.g., drone can fly over gaps; human requires stable ground).
- Distinguish traversal affordances: what the drone can traverse vs what a human can traverse.
- Do NOT change the recommendation.
- Do not invent hazards not present in the input list. Only reference hazards explicitly provided.
- Follow the structured format exactly.
- If fallback_available is True, explain that a previously observed safe state may be preferable.
- Do NOT specify spatial directions, distances, or path planning."""

    messages = [
        SYS_EXPLAIN,
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    cfg = get_config().cosmos
    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens_explanation,
            do_sample=cfg.generation.do_sample,
        )

    if cfg.timing:
        print("Cosmos explanation latency:", round(time.time() - t0, 2), "s")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return decoded.strip()
