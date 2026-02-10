"""
Cosmos Reason 2 smoke test — sanity proof that the project can invoke Reason 2
and get a physically grounded response on one image.

Prerequisites: Run inside the Cosmos Reason 2 venv (or an env with transformers,
torch, PIL). Cosmos repo at ~/cosmos-reason2 unless COSMOS_REASON2_ROOT is set.

Usage:
  python smoke_test.py                    # uses scripts/test_image.png
  python smoke_test.py path/to/image.jpg # use a specific image
"""

import os
import sys
from pathlib import Path

# --- Make Cosmos Reason 2 importable ---
COSMOS_ROOT = Path(os.environ.get("COSMOS_REASON2_ROOT", "")) or (
    Path.home() / "cosmos-reason2"
)
sys.path.insert(0, str(COSMOS_ROOT))

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch


def main():
    print("=== Cosmos Reason 2 Smoke Test ===")

    # --- Resolve image path: CLI arg or default next to this script ---
    script_dir = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1]).resolve()
    else:
        image_path = script_dir / "test_image.png"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found at {image_path}. "
            "Add a simple test image (e.g. scripts/test_image.png) or pass a path: "
            "python smoke_test.py path/to/image.jpg"
        )

    # --- Load model & processor ---
    model_id = "nvidia/Cosmos-Reason2-2B"

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # --- Chat-style prompt with image inside message (required by Qwen3-VL) ---
    # Image must be in the message content; use apply_chat_template only (no separate images=).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": (
                        "You are a drone reasoning about shared physical safety.\n"
                        "Analyze the scene and answer:\n"
                        "1. What physical hazards are visible?\n"
                        "2. Is the path ahead safe for the drone?\n"
                        "3. Is the same path safe for a human to follow?\n"
                        "Answer concisely using physical reasoning."
                    ),
                },
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

    print("Running inference...")
    generated_ids = model.generate(**inputs, max_new_tokens=300)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n=== MODEL OUTPUT ===")
    print(output_text[0])
    print("====================")


if __name__ == "__main__":
    main()
