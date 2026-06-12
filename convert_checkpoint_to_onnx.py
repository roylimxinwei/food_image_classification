"""Convert PyTorch Lightning checkpoint to ONNX format."""

import torch
import sys
import os
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent / "food_classifier_project"))
from utils import FoodClassifier, Config

# Configuration
CHECKPOINT_PATH = r"food_classifier_project\checkpoints\sgfood233_convnext_base\epoch=09-val_acc=0.8345.ckpt"
OUTPUT_PATH = r"sgfood233_convnext_base.onnx"
MODEL_NAME = "convnext_base"
NUM_CLASSES = 233

def convert_to_onnx():
    """Load checkpoint and export to ONNX."""

    # Load the trained model from checkpoint
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    model = FoodClassifier.load_from_checkpoint(
        CHECKPOINT_PATH,
        backbone_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
    )

    # Set to evaluation mode
    model.eval()
    model.cpu()

    # Create dummy input
    dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    print(f"Exporting to ONNX: {OUTPUT_PATH}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"SUCCESS: ONNX model saved!")
    print(f"  Location: {os.path.abspath(OUTPUT_PATH)}")
    print(f"  Size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")

    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(OUTPUT_PATH)
        onnx.checker.check_model(onnx_model)
        print("SUCCESS: ONNX model verification passed!")
    except ImportError:
        print("  Note: Install 'onnx' package to verify the model")
    except Exception as e:
        print(f"  Warning: ONNX verification failed: {e}")

if __name__ == "__main__":
    convert_to_onnx()
