# Fine-tune EfficientNet

Fine-tune a pre-trained **TF-EfficientNet-Lite4** model on a custom image dataset.

1. Create a `dataset` folder.
2. Create subfolders (example: dog, cat, fish, bird) and populate them.
3. Run `python src/fine_tune_from_base.py`.

When its finished it will return a checkpoint file, labels.txt, and the fine-tuned model.