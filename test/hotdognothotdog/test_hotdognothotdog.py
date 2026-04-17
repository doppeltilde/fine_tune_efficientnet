from pathlib import Path
import mediapipe as mp
import numpy as np
from PIL import Image

BaseOptions = mp.tasks.BaseOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "test/hotdognothotdog/model.tflite"
image_paths = ["test/hotdognothotdog/hotdog.jpg", "test/hotdognothotdog/nothotdog.jpg"]

options = ImageClassifierOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE,
)

with ImageClassifier.create_from_options(options) as classifier:
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((380, 380), Image.Resampling.LANCZOS)
        numpy_image = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        classification_result = classifier.classify(mp_image)

        for classification in classification_result.classifications:
            print(image_path)
            for category in classification.categories:
                print(f"{category.category_name}: {category.score:.10f}")
