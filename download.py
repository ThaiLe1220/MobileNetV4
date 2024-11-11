import kagglehub

import shutil
import os

# Download latest version
path = kagglehub.dataset_download("alxmamaev/flowers-recognition")

print("Path to dataset files:", path)
destination_path = "/home/ubuntu/Desktop/eugene/MobileNetV4"

# Create the destination directory if it doesn't exist
os.makedirs(destination_path, exist_ok=True)

# Copy the contents of the source directory to the destination
shutil.copytree(path, destination_path, dirs_exist_ok=True)

print(f"Dataset moved to: {destination_path}")
