import torch
import torchvision
import torchvision.transforms as transforms
import timm
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import ast

# Print library versions and CUDA status
print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"timm version: {timm.__version__}")
# print(timm.list_models())
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Load ImageNet class labels
with open("imagenet1000_clsidx_to_labels.txt", encoding="utf-8") as f:
    image_net_labels = [line.strip() for line in f.readlines()]
    # labels_dict = ast.literal_eval(f.read())

# image_net_labels = [labels_dict[i] for i in range(len(labels_dict))]


def unnormalize(tensor, mean, std):
    """
    Reverse the normalization of a tensor image.

    Args:
        tensor (torch.Tensor): Normalized tensor image of shape (C, H, W)
        mean (list): Mean used for normalization
        std (list): Standard deviation used for normalization

    Returns:
        torch.Tensor: Unnormalized tensor image
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_tensor(tensor, mean, std, title=None):
    """
    Visualize a tensor as an image.

    Args:
        tensor (torch.Tensor): Tensor image of shape (C, H, W)
        mean (list): Mean used for normalization
        std (list): Standard deviation used for normalization
        title (str, optional): Title of the plot
    """
    tensor = unnormalize(tensor.clone(), mean, std)
    tensor = torch.clamp(tensor, 0, 1)  # Ensure the values are within [0,1]
    np_img = tensor.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))  # Convert to HWC format for matplotlib
    plt.imshow(np_img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def predict_with_ImageNetV4(pil_img):
    """
    Perform inference using a MobileNetV4 model and visualize the transformed image tensor.

    Args:
        pil_img (PIL.Image.Image): Input image as a PIL object
    """
    # Specify the model name
    # model_name = "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k" # top1: 84.99, top5: 97.294, param_count: 32.59M, img_size: 544, normal: 131MB, onnx: 130MB
    # model_name = "mobilenetv4_hybrid_large.ix_e600_r384_in1k" # top1: 84.356, top5: 96.892, param_count: 37.76M, img_size: 448, normal: 152MB
    model_name = "mobilenetv4_hybrid_medium.ix_e550_r384_in1k"  # top1: 83.394, top5: 96.760, param_count: 11.07M, img_size: 448, normal: 44.7MB

    # Load the pretrained MobileNetV4 model from timm
    try:
        model = timm.create_model(model_name, pretrained=True)
    except Exception as e:
        raise ValueError(f"Error loading model '{model_name}': {e}")
    model.eval()

    # Get model-specific transforms (normalization, resize, etc.)
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config)

    # Apply the transforms and add batch dimension
    input_tensor = transform(pil_img).unsqueeze(0)

    # TODO: Visualize the input_tensor after transforming
    # Extract mean and std from data_config for unnormalization
    mean = data_config["mean"]
    std = data_config["std"]

    # Remove batch dimension for visualization
    tensor_to_visualize = input_tensor[0]
    visualize_tensor(tensor_to_visualize, mean, std, title="Transformed Image Tensor")

    # Move model and tensor to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top-5 probabilities and class indices
    top5_probabilities, top5_class_indices = torch.topk(probabilities, k=5)

    # Convert probabilities to percentages
    top5_probabilities = top5_probabilities.cpu().numpy() * 100
    top5_class_indices = top5_class_indices.cpu().numpy()

    # Map the class indices to the related word in ImageNet labels
    predictions = []
    for i in range(5):
        idx = top5_class_indices[i]
        if 0 <= idx < len(image_net_labels):
            # label = image_net_labels[idx]
            label_line = image_net_labels[idx]
            label = label_line.split(": ")[1].strip().strip("',")
        else:
            label = "Unknown"
        prob = round(top5_probabilities[i], 2)
        predictions.append([label, prob])

    # Print the top-5 predictions
    print("\nTop-5 Predictions:")
    for pred in predictions:
        print(f"{pred[0]}: {pred[1]}%")


# Path to the input image
image_path = "images/image2.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image file not found at {image_path}. Please ensure the path is correct."
    )

# Load the image as a PIL image and ensure it's in RGB format
pil_image = Image.open(image_path).convert("RGB")

# Perform prediction and visualization
predict_with_ImageNetV4(pil_image)
