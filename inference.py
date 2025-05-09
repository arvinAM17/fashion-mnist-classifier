# inference.py
import argparse
from pathlib import Path
import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.models import FashionMNISTBaseline, FashionMNISTCNN
from src.utils.load_config import load_config

# Set up labels for FashionMNIST classes
LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Load config
config = load_config()
device = config["device"] if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_name = config["model_name"]
epochs = config["epochs"]

model_map = {
    "fc": lambda: FashionMNISTBaseline(hidden_layers=config["hidden_dim"], output_dimension=config["output_dim"]),
    "cnn": lambda: FashionMNISTCNN(hidden_layer_dimension=config["hidden_dim"], output_dimension=config["output_dim"]),
}
model = model_map.get(model_name, model_map["cnn"])().to(device)

# Load weights
model_path = Path("models") / f"{model_name}_fashionmnist_epoch{epochs}.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])

def load_image(image_path):
    image = Image.open(image_path).convert("L")  # ensure grayscale
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

def predict(tensor):
    with torch.no_grad():
        output = model(tensor)
        predicted_idx = output.argmax(dim=1).item()
        return predicted_idx

def main(image_path=None):
    if image_path:
        image_tensor = load_image(image_path)
    else:
        # Load sample from test dataset
        test_dataset = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
        image_tensor, label = test_dataset[np.random.randint(0, len(test_dataset) - 1)]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        print(f"Actual label: {LABELS[label]}")

    predicted_idx = predict(image_tensor)
    print(f"Predicted label: {LABELS[predicted_idx]}")

    # Show image
    image = image_tensor.squeeze().cpu().numpy()
    plt.imshow(image, cmap="gray")
    plt.title(f"Prediction: {LABELS[predicted_idx]}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file")
    args = parser.parse_args()
    main(args.image)
    exit()
