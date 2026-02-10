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
from PIL import Image


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

    image = Image.open(image_path).convert("RGB")

    # --- Load model & processor ---
    model_id = "nvidia/Cosmos-Reason2-2B"

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # --- Chat-style prompt with explicit image placeholder (required by Qwen3-VL) ---
    # Without {"type": "image"} the model gets 0 image tokens but 1536 image features → ValueError.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
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

    inputs = processor(
        messages,
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    print("Running inference...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300
    )

    result = processor.decode(outputs[0], skip_special_tokens=True)

    print("\n=== MODEL OUTPUT ===")
    print(result)
    print("====================")


if __name__ == "__main__":
    main()
