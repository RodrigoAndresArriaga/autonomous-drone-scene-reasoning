# Cosmos Reason 2 interface: structured perception and strategic explanation.
# Layer 1: schema-constrained hazard extraction.
# Layer 3: explanation of deterministic outcomes.

import json
import os
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .hazard_schema import HAZARD_TYPES

_model = None
_processor = None

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


def _load_model():
    global _model, _processor
    if _model is None:
        model_name = "nvidia/Cosmos-Reason2-2B"
        _processor = AutoProcessor.from_pretrained(model_name)
        try:
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except Exception:
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa",
            )
        if torch.cuda.is_available():
            dev = next(_model.parameters()).device
            if "cpu" in str(dev):
                import warnings
                warnings.warn(f"Model loaded on {dev}; expected cuda. Check device_map.")
        if os.environ.get("COSMOS_TIMING"):
            attn = getattr(_model.config, "_attn_implementation", None) or getattr(_model.config, "attn_implementation", "?")
            print("Attention implementation:", attn)
    return _model, _processor

# Layer 1: Structured hazard extraction from image. Returns schema-constrained hazards + visibility_status. Cosmos does NOT output safety or recommendations. Cosmos only extracts hazards from the image.
def query_cosmos_structured(image_path: str) -> dict:
    model, processor = _load_model()

    img = Image.open(image_path).convert("RGB")
    img = img.resize((448, 448))

    allowed_hazards = ", ".join(HAZARD_TYPES.keys())

    prompt = f"""Analyze the image and return ONLY valid JSON in the following format:

{{
  "hazards": [
    {{"type": "<hazard_type>", "severity": "<low|medium|high|critical|contextual>"}}
  ],
  "visibility_status": "<clear|occluded|unknown>"
}}

Allowed hazard types: {allowed_hazards}

Rules:
- Each hazard's "type" field must be exactly one value from the allowed list (e.g. "hole", "debris").
- You may include multiple hazards in the array.
- Do NOT modify hazard type names. Do NOT add prefixes or invent new categories.
- If a scene element resembles a hazard, map it to the closest allowed type.
- Output valid JSON only. No commentary.
- Use compact JSON: no extra newlines, indentation, or whitespace (e.g. {{"hazards":[{{"type":"hole","severity":"critical"}}],"visibility_status":"clear"}}).

Example mappings:
- Collapsed staircase → hole
- Construction debris pile → unstable_debris_stack
- Blocked path → restricted_escape_route
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
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
    t1 = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True,
        )
    t2 = time.time()

    if os.environ.get("COSMOS_TIMING"):
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

    try:
        start = decoded.index("{")
        end = decoded.rindex("}") + 1
        json_str = decoded[start:end]
        parsed = json.loads(json_str)
    except Exception as e:
        if os.environ.get("COSMOS_TIMING"):
            print("JSON parse failed:", e)
            print("Decoded snippet:", repr(decoded[:300] + "..." if len(decoded) > 300 else decoded))
        return {"hazards": [], "visibility_status": "unknown"}

    return parsed

# Layer 3: Strategic explanation. Cosmos receives hazards, safety, recommendation and produces clear explanation, human instructions, tactical justification. Cosmos explains the deterministic outcome; it does not compute it.
def query_cosmos_explanation(context: dict) -> str:
    model, processor = _load_model()

    prompt = f"""Explain the deterministic safety decision below.

Hazards: {context.get("hazards", [])}
Drone safety: {context.get("safety", {}).get("drone_path_safety")}
Human safety: {context.get("safety", {}).get("human_follow_safety")}
Recommendation: {context.get("recommendation", {}).get("recommendation")}

Rules:
- Do NOT change the recommendation.
- Do NOT introduce new hazards.
- Provide clear reasoning for why this decision is appropriate."""

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    if os.environ.get("COSMOS_TIMING"):
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
