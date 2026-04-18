from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from pathlib import Path
import traceback
import flatbuffers


def main():
    tflite_path = "output/model.tflite"
    labels_path = Path("output/labels.txt")
    print("\nAttaching metadata for float32 ViT model...")

    try:
        attach_metadata(tflite_path, labels_path)
        print("Metadata attached successfully.")
    except Exception as e:
        print(f"Metadata failed: {e}")
        traceback.print_exc()


def attach_metadata(tflite_path: str, labels_path: Path):
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "ViT-Base-Patch16-224 Image Classifier"
    model_meta.version = "1.0"
    model_meta.description = (
        "Fine-tuned Vision Transformer."
        "Normalization (ImageNet mean/std) is applied internally."
    )

    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = (
        "Input RGB image, float32, pixel values in [0, 255]. "
        "Internal preprocessing: divide by 255, subtract ImageNet mean [0.485, 0.456, 0.406], "
        "divide by std [0.229, 0.224, 0.225]."
    )

    norm_unit = _metadata_fb.ProcessUnitT()
    norm_unit.optionsType = _metadata_fb.ProcessUnitOptions.NormalizationOptions
    norm_opts = _metadata_fb.NormalizationOptionsT()
    norm_opts.mean = [
        0.485 * 255.0,
        0.456 * 255.0,
        0.406 * 255.0,
    ]
    norm_opts.std = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
    norm_unit.options = norm_opts
    input_meta.processUnits = [norm_unit]

    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Softmax class probabilities"

    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = labels_path.name
    label_file.description = "Class labels"
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER,
    )
    metadata_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_file(tflite_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([str(labels_path)])
    populator.populate()

    print(f"Metadata written to: {tflite_path}")


if __name__ == "__main__":
    main()
