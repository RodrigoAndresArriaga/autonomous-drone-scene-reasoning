# Cosmos Reason 2 interface: structured perception and strategic explanation.
# Layer 1: schema-constrained hazard extraction.
# Layer 3: explanation of deterministic outcomes.

import json
import logging
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from configs.config import get_config
from safety.affordance_model import SEVERITY_WEIGHTS
from safety.recommendation import RECOMMENDATIONS
from .hazard_schema import HAZARD_TYPES

_FALLBACK_RECOMMENDATION = "Reroute (previously observed safe state available)"

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
    json_str = re.sub(r'"([^"]+)"\s*"visibility_status"', r'"\1", "visibility_status"', json_str)
    # Remove visibility_status from inside hazard objects (schema expects only at top level)
    json_str = re.sub(r',\s*"visibility_status"\s*:\s*"[^"]*"', "", json_str)
    json_str = re.sub(r'"visibility_status"\s*:\s*"[^"]*"\s*,?\s*', "", json_str)
    # Fix trailing commas before ] or }
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    return json_str


# Raw type (from Layer 1) -> canonical type for fallback when Layer 2 parse fails
_RAW_TO_CANONICAL = {
    "fire": "heat_source_proximity",
    "flames": "heat_source_proximity",
    "barbed wire": "entanglement_risk",
    "razor wire": "entanglement_risk",
    "hanging cable": "entanglement_risk",
    "hanging wires": "entanglement_risk",
    "suspended cables": "entanglement_risk",
    "collapsed staircase": "partial_floor_collapse",
    "collapsed floor": "partial_floor_collapse",
    "collapsed wooden beams": "partial_floor_collapse",
    "collapsed beams": "partial_floor_collapse",
    "hole": "hole",
    "excavation": "hole",
    "collapsed furniture": "unstable_debris_stack",
    "collapsed couch": "unstable_debris_stack",
    "collapsed sofa": "unstable_debris_stack",
    "collapsed fireplace": "unstable_debris_stack",
    "collapsed television": "unstable_debris_stack",
    "collapsed television stand": "unstable_debris_stack",
    "collapsed television screen": "unstable_debris_stack",
    "debris": "unstable_debris_stack",
    "electrical": "electrical_exposure",
    "wiring": "electrical_exposure",
    "exposed wiring": "electrical_exposure",
    "low ceiling": "narrow_passage",
    "overhang": "narrow_passage",
    "blocked doorway": "restricted_escape_route",
    "blocked path": "restricted_escape_route",
    "hanging debris": "overhead_instability",
    "collapsing ceiling": "overhead_instability",
}


def _map_raw_to_canonical(raw_type: str) -> str | None:
    """Map raw hazard type to canonical HAZARD_TYPES key, or None if no mapping."""
    t = raw_type.strip().lower()
    if t in _RAW_TO_CANONICAL:
        canonical = _RAW_TO_CANONICAL[t]
        return canonical if canonical in HAZARD_TYPES else None
    if "collapsed" in t and ("stair" in t or "floor" in t or "beam" in t):
        return "partial_floor_collapse"
    if "collapsed" in t or "debris" in t:
        return "unstable_debris_stack"
    if "fire" in t or "flame" in t:
        return "heat_source_proximity"
    if ("hanging" in t and ("debris" in t or "ceiling" in t)) or "collapsing" in t:
        return "overhead_instability"
    if "barbed" in t or "razor" in t or "wire" in t or "cable" in t or "hanging" in t:
        return "entanglement_risk"
    if "hole" in t or "excavation" in t:
        return "hole"
    if "electrical" in t or "wiring" in t:
        return "electrical_exposure"
    if "low ceiling" in t or "overhang" in t:
        return "narrow_passage"
    if "blocked" in t and ("door" in t or "path" in t or "exit" in t):
        return "restricted_escape_route"
    return None


def extract_scene_summary_from_layer1(raw_text: str) -> str:
    """Extract scene_summary from Layer 1 raw output. Layer 2 may not pass it through."""
    if not raw_text or not raw_text.strip():
        return ""
    json_str = _preprocess_json(raw_text)
    try:
        parsed = json.loads(json_str)
        return str(parsed.get("scene_summary", "")).strip()
    except json.JSONDecodeError:
        pass
    m = re.search(r'"scene_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).replace("\\n", "\n").strip()
    return ""


def _fallback_extract_from_raw(raw_text: str) -> dict:
    """When Layer 2 parse fails, extract hazards from raw Layer 1 text via regex and map to canonical."""
    json_str = _preprocess_json(raw_text)
    json_str = _try_repair_normalize_json(json_str)
    hazards = []
    try:
        parsed = json.loads(json_str)
        for h in parsed.get("hazards", []) or []:
            if isinstance(h, dict) and "type" in h:
                canonical = _map_raw_to_canonical(str(h["type"]))
                if canonical and canonical in HAZARD_TYPES:
                    hout = {"type": canonical, "severity": str(h.get("severity", "medium")), "zone": "unknown"}
                    hazards.append(hout)
        return {"hazards": hazards, "visibility_status": "unknown", "scene_summary": parsed.get("scene_summary", "")}
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r'"type"\s*:\s*"([^"]+)"\s*,\s*"severity"\s*:\s*"([^"]+)"', json_str, re.IGNORECASE):
        raw_type, severity = m.group(1), m.group(2)
        canonical = _map_raw_to_canonical(raw_type)
        if canonical:
            hazards.append({"type": canonical, "severity": severity, "zone": "unknown"})
    return {"hazards": hazards, "visibility_status": "unknown", "scene_summary": ""}


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
        # transformers load_video expects a local path or http(s) URL, not file:// URI
        return {"type": "video", "video": abs_path, "fps": fps}
    raise ValueError(f"Unknown media_type: {media_type!r}, must be 'image' or 'video'")


# Layer 1: Structured hazard extraction from image or video. Returns schema-constrained hazards + visibility_status.
# Cosmos does NOT output safety or recommendations. Cosmos only extracts hazards from the media.
def query_cosmos_extract(media_path: str, media_type: str = "image", fps: int = 4) -> dict:
    model, processor = _load_model()

    media_item = _build_media_content(media_path, media_type, fps)

    prompt = """You are a structured hazard extraction engine for physical safety reasoning.

Analyze the provided media (image or video clip) and return ONLY valid JSON.

First, provide a brief scene_summary (1-3 sentences) describing what is visibly present.

Then list hazards.

Format:
{
  "scene_summary": "<natural description of what is visible>",
  "hazards": [
    {"type": "<hazard_description>", "severity": "<low|medium|high|critical|contextual>"}
  ],
  "visibility_status": "<clear|occluded|unknown>"
}

If no hazards are visible, return: {"scene_summary":"<description>","hazards":[],"visibility_status":"clear"}
If uncertain, use "unknown" for visibility_status.

Rules:
- scene_summary must describe only what is visible. It must not include recommendations or safety advice.
- Hazards MUST include a severity field.
- Severity MUST be one of: low, medium, high, critical, contextual.
- Do NOT omit severity under any circumstance.
- Use descriptive hazard names (e.g. incomplete stairs, debris pile, collapsed staircase).
- Output valid JSON only. No commentary.

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
        print("[Layer 1] Input prep:", round(t1 - t0, 2), "s | Generate:", round(t2 - t1, 2), "s | Total:", round(t2 - t0, 2), "s")
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

    hazard_ontology = "\n".join(
        f"- {k}: default severity={v.get('severity', 'medium')}"
        for k, v in HAZARD_TYPES.items()
    )

    prompt = f"""You are a hazard taxonomy normalizer.

The following is raw output from a hazard extraction model that analyzed an image or video.
Extract all hazards mentioned and map each to the closest canonical hazard type.

Raw Layer 1 output:
{raw_extraction}

Canonical hazard taxonomy (from HAZARD_TYPES):
{hazard_ontology}

For each hazard:
- Select the single most appropriate canonical type from the allowed list.
- The "type" field MUST exactly match one of the canonical type keys above.
- Do NOT echo raw names.
- If the raw description is more specific than the canonical type, map it to the closest broader canonical category.

Zone: one of ground | mid | overhead | unknown
- ground = on floor or affecting foot support
- mid = body-level obstacle in traversal space
- overhead = above head or in airspace
- unknown = cannot determine

Severity:
- Preserve the original severity if provided.
- If no severity is present, use the canonical default severity.
- For restricted_escape_route, always use severity "contextual".

Rules:
- Never invent hazards.
- Never output a type not in the canonical taxonomy above.
- If a hazard does not clearly match any canonical type, do not output it. (Conservative omission — code fallback will catch common cases; avoid creative taxonomy stretching.)
- Assign a zone for every hazard.
- Use "unknown" zone if uncertain.
- Extract visibility_status if present in the raw output; otherwise use "unknown".

Return ONLY valid compact JSON:
{{"hazards":[{{"type":"<canonical>","severity":"<low|medium|high|critical|contextual>","zone":"<ground|mid|overhead|unknown>"}}],"visibility_status":"<clear|occluded|unknown>"}}"""

    messages = [
        SYS_NORMALIZE,
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    t0 = time.time()
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
    # Apply repair before first parse (Layer 2 often echoes malformed JSON from Layer 1)
    json_str = _try_repair_normalize_json(json_str)
    if cfg.timing:
        print("Layer 2 JSON (after repair):", repr(json_str[:800] + "..." if len(json_str) > 800 else json_str))
    parsed = None
    first_error = None
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        first_error = e
        if cfg.timing:
            print("Layer 2 JSON parse failed:", first_error)
            print("Layer 2 decoded (raw):", repr(decoded[:500] + "..." if len(decoded) > 500 else decoded))
    if parsed is None:
        global _normalization_parse_failures
        _normalization_parse_failures += 1
        if cfg.timing:
            print("[Layer 2] Parse failed, using fallback extract from raw Layer 1")
        return _fallback_extract_from_raw(raw_extraction)

    raw_hazards = parsed.get("hazards", [])
    filtered = []
    for h in raw_hazards:
        if not isinstance(h, dict):
            continue
        htype = h.get("type")
        if htype not in HAZARD_TYPES:
            mapped = _map_raw_to_canonical(str(htype))
            if mapped and mapped in HAZARD_TYPES:
                htype = mapped
            else:
                continue
        h["type"] = htype  # ensure consistency downstream (zone/severity stay aligned with canonical type)
        severity = h.get("severity", "medium")
        if htype == "restricted_escape_route":
            severity = "contextual"
        zone = h.get("zone", "unknown")
        if zone not in ("ground", "mid", "overhead", "unknown"):
            zone = "unknown"
        filtered.append({"type": htype, "severity": severity, "zone": zone})

    # Collapse by (type, zone), keep highest severity per group (reduces log noise, prompt tokens)
    dedup: dict[tuple[str, str], dict] = {}
    for h in filtered:
        key = (h["type"], h.get("zone", "unknown"))
        if key not in dedup:
            dedup[key] = h
        else:
            w = SEVERITY_WEIGHTS.get(h.get("severity", "medium"), 2)
            dw = SEVERITY_WEIGHTS.get(dedup[key].get("severity", "medium"), 2)
            if w > dw:
                dedup[key] = h
    filtered = list(dedup.values())

    if cfg.timing:
        print("Layer 2 hazards (before filter):", raw_hazards)
        dropped = [h.get("type") for h in raw_hazards if isinstance(h, dict) and h.get("type") not in HAZARD_TYPES]
        if dropped:
            print("Layer 2 hazard types dropped (not in HAZARD_TYPES):", dropped)
        print("Layer 2 hazards (after collapse):", filtered)
    parsed["hazards"] = filtered
    parsed["scene_summary"] = parsed.get("scene_summary", "")
    vs = parsed.get("visibility_status", "unknown")
    parsed["visibility_status"] = vs if vs in ("clear", "occluded", "unknown") else "unknown"
    if cfg.timing:
        print("[Layer 2] Normalize total:", round(time.time() - t0, 2), "s")
    return parsed


def _postprocess_explanation(text: str, rec_text: str) -> str:
    """Fix section headers and ensure Decision matches canonical recommendation."""
    # Normalize section headers (case-insensitive, allow optional extra spaces)
    headers = [
        ("Scene Context:", r"Scene\s*Context\s*:"),
        ("Hazards:", r"Hazards\s*:"),
        ("Drone Safety:", r"Drone\s*Safety\s*:"),
        ("Human Safety:", r"Human\s*Safety\s*:"),
        ("Decision:", r"Decision\s*:"),
        ("Justification:", r"Justification\s*:"),
    ]
    for canonical, pattern in headers:
        text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)

    # Replace Decision section content with exact rec_text
    match = re.search(r"(Decision:\s*\n)([^\n]+)", text)
    if match and match.group(2).strip() != rec_text:
        text = text[: match.start(2)] + rec_text + text[match.end(2) :]
    return text


# Layer 3: Strategic explanation. Cosmos receives hazards, safety, recommendation and produces clear explanation, human instructions, tactical justification. Cosmos explains the deterministic outcome; it does not compute it.
def query_cosmos_explanation(context: dict) -> str:
    model, processor = _load_model()

    raw_extraction = context.get("raw_extraction") or "N/A"
    scene_summary = context.get("scene_summary") or ""
    if os.environ.get("COSMOS_DEBUG") == "1":
        print("DEBUG scene_summary:", (scene_summary or "")[:120])
    hazards = context.get("hazards", [])
    rec = context.get("recommendation", {})
    # Recommendation from safety.recommendation.generate_navigation_recommendation (called in scene_agent, passed via context).
    # Canonical values in RECOMMENDATIONS; scene_agent may override to fallback when previously observed safe state exists.
    rec_text = rec.get("recommendation", "") if isinstance(rec, dict) else str(rec)
    _allowed = set(RECOMMENDATIONS.values()) | {_FALLBACK_RECOMMENDATION}
    if rec_text and rec_text not in _allowed:
        logging.warning("Layer 3: rec_text %r not in canonical RECOMMENDATIONS", rec_text)
    if os.environ.get("COSMOS_TIMING") == "1":
        print(f"[Layer 3] Received OUR computed recommendation: \"{rec_text}\"")
        print("[Layer 3] LLM will EXPLAIN this decision only. OUR computation drives the Decision section; LLM does not decide.")
    safety = context.get("safety", {})

    hazards_narrative = ", ".join(
        f"{h.get('type', '?')} ({h.get('severity', 'medium')}, zone={h.get('zone', 'unknown')})"
        for h in hazards
    ) if hazards else "None observed"
    drone_safety = safety.get("drone_path_safety", {})
    human_safety = safety.get("human_follow_safety", {})
    drone_class = drone_safety.get("classification", "unknown")
    drone_risk = drone_safety.get("total_risk_score", 0)
    human_class = human_safety.get("classification", "unknown")
    human_risk = human_safety.get("total_risk_score", 0)
    drone_violated = drone_safety.get("violated_constraints", [])
    human_violated = human_safety.get("violated_constraints", [])

    drone_narrative = f"{drone_class} (risk {drone_risk})" + (
        f", violated: {', '.join(drone_violated)}" if drone_violated else ""
    )
    human_narrative = f"{human_class} (risk {human_risk})" + (
        f", violated: {', '.join(human_violated)}" if human_violated else ""
    )

    prompt = f"""Explain the deterministic safety decision below.

Observed scene summary:
{scene_summary or "(none provided)"}

Hazards (type, severity, zone): {hazards_narrative}
Drone safety: {drone_narrative}
Human safety: {human_narrative}
Recommendation: {rec_text}
Deterministic recommendation (do not change): {rec_text}
Fallback available: {context.get("fallback_available", False)}

Agent constraints:
Drone:
- Can fly over gaps
- Does not require stable ground
- Requires clear airspace (mid, overhead)
- Sensitive to electricity, heat, entanglement
- Exposed to falling debris

Human:
- Requires stable ground
- Requires body clearance (narrow passages, low ceiling)
- Cannot fly over gaps
- Sensitive to instability, collapse, confined air
- Exposed to falling debris

You MUST write natural language prose in each section. Do NOT copy, echo, or paste the raw data structures (hazards, drone_safety, human_safety) into your response. Explain in sentences that a human can read.

Respond in this exact structure. Each section must start with the header and a newline:

Scene Context:
<brief reference to observed scene_summary>

Hazards:
<what is physically present>

Drone Safety:
<what constraints are violated and why>

Human Safety:
<what constraints are violated and why>

Decision:
<state the final deterministic recommendation exactly as given>

Justification:
<3–5 sentences maximum. Do not repeat content. Stop after 5 sentences.>

Rules:
- Each section (Scene Context, Hazards, Drone Safety, Human Safety, Decision, Justification) must contain explanatory prose. Do NOT output raw dicts or JSON.
- Use physically grounded language. Refer to specific hazards from the perception summary.
- Reference zone and violated constraints in your explanation to show embodied reasoning.
- Explicitly mention agent constraints (e.g., drone can fly over gaps; human requires stable ground).
- Distinguish traversal affordances: what the drone can traverse vs what a human can traverse.
- The Decision section must restate the exact recommendation provided. The Decision section must exactly match the text from "Deterministic recommendation (do not change)" above.
- The Justification must explicitly connect violated constraints to that decision.
- The Justification section must contain at most 5 sentences.
- Do not repeat any sentence within or across sections.
- End immediately after the justification. Do not continue generating.
- Do not reinterpret or change the recommendation.
- Do not invent hazards not present in the input. Only reference hazards explicitly provided.
- Follow the structured format exactly.
- Use exact section headers including the space: "Drone Safety:", "Human Safety:".
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
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

    if cfg.timing:
        print("[Layer 3] Explanation:", round(time.time() - t0, 2), "s")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    decoded = _postprocess_explanation(decoded, rec_text)
    return decoded.strip()
