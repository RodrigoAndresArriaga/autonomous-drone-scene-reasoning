# Cosmos Reason 2 interface: structured perception and strategic explanation.
# Layer 1: schema-constrained hazard extraction.
# Layer 3: explanation of deterministic outcomes.

import json
import torch
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
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    return _model, _processor

# Layer 1: Structured hazard extraction from image. Returns schema-constrained hazards + visibility_status. Cosmos does NOT output safety or recommendations. Cosmos only extracts hazards from the image.
def query_cosmos_structured(image_path: str) -> dict:
    model, processor = _load_model()

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
- Only use hazard types from the allowed list.
- Output JSON only. No explanation.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
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

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

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
    except Exception:
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

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

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
