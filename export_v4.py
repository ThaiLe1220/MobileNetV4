import os
import shutil


def convert_to_onnx(
    model_name: str = "mobilenetv4_hybrid_medium.ix_e550_r384_in1k",
    onnx_filename: str = "mobilenetv4.onnx",
    input_size: tuple = (1, 3, 448, 448),
):
    """
    Converts a PyTorch model to ONNX format.
    """
    try:
        import torch
        import timm

        # Load the pretrained model
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        # Create a dummy input with the correct input size
        dummy_input = torch.randn(*input_size)

        # Export the model to ONNX format
        torch.onnx.export(
            model,
            dummy_input,
            onnx_filename,
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
        )
        print(f"Model successfully converted to ONNX and saved as {onnx_filename}")
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")


def convert_onnx_to_tf(
    onnx_filename: str = "mobilenetv4.onnx",
    saved_model_dir: str = "mobilenetv4_saved_model",
):
    """
    Converts an ONNX model to TensorFlow SavedModel format.
    """
    try:
        import onnx
        from onnx_tf.backend import prepare

        # Load the ONNX model
        onnx_model = onnx.load(onnx_filename)
        print(f"Loaded ONNX model from {onnx_filename}")

        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)

        # Remove the directory if it already exists to avoid conflicts
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)

        # Export the TensorFlow SavedModel
        tf_rep.export_graph(saved_model_dir)
        print(
            f"Model successfully converted to TensorFlow SavedModel and saved in {saved_model_dir}"
        )
    except Exception as e:
        print(f"Error during ONNX to TensorFlow conversion: {e}")


def convert_tf_to_tflite(
    saved_model_dir: str = "mobilenetv4_saved_model",
    tflite_filename: str = "mobilenetv4.tflite",
):
    """
    Converts a TensorFlow SavedModel to TensorFlow Lite format.
    """
    try:
        import tensorflow as tf

        # Create the TFLite converter
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

        # Optimize the model (optional but recommended)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert the model
        tflite_model = converter.convert()

        # Save the TFLite model
        with open(tflite_filename, "wb") as f:
            f.write(tflite_model)
        print(
            f"Model successfully converted to TensorFlow Lite and saved as '{tflite_filename}'"
        )
    except Exception as e:
        print(f"Error during TensorFlow to TFLite conversion: {e}")


if __name__ == "__main__":
    # Step 1: Convert PyTorch model to ONNX [Worked]
    # convert_to_onnx()

    # Step 2: Convert ONNX to TensorFlow SavedModel [Error]
    convert_onnx_to_tf()

    # Step 3: Convert TensorFlow SavedModel to TensorFlow Lite
    # convert_tf_to_tflite()

    print("hello")
