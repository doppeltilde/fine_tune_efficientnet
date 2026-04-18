import os
import torch
import timm
import litert_torch
from pathlib import Path
import traceback


class ProbabilityModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        logits = self.model(x)
        return torch.nn.functional.softmax(logits, dim=1)


def main():
    print("Loading fine-tuned Vision Transformer model for conversion...")

    checkpoint_path = "output/vit_base_patch16_224_final_finetuned.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    print(f"Loading model with {num_classes} classes from checkpoint.")

    base_model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )

    base_model.load_state_dict(checkpoint["model_state_dict"])
    print("Fine-tuned ViT model loaded successfully.")

    labels_path = Path("output/labels.txt")
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    with open(labels_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"labels.txt created successfully at: {labels_path}")
    print(f"Total labels written: {len(class_names)}")

    model = ProbabilityModel(base_model)
    model.eval()

    print("\nConverting Vision Transformer model to TensorFlow Lite format...")

    try:
        sample_input = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.float32)

        edge_model = litert_torch.convert(model, (sample_input,))

        os.makedirs("output", exist_ok=True)
        tflite_path = "output/model.tflite"

        edge_model.export(tflite_path)

        print(f"Conversion successful!")
        print(f"TensorFlow Lite model saved to: {tflite_path}")

    except Exception as e:
        print(f"\nLiteRT conversion failed: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
