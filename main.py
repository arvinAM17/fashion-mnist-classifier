# main.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.load_config import load_config
from src.models import FashionMNISTCNN, FashionMNISTBaseline
from src.training.train_loop import train_model, evaluate
from src.utils.device import get_best_device

from pathlib import Path


# Load config
config = load_config()

device = get_best_device(config["device"])

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

model_name = config["model_name"]
epochs = config["epochs"]

model_map = {
    "fc": lambda: FashionMNISTBaseline(hidden_layers=config["hidden_dim"], output_dimension=config["output_dim"]),
    "cnn": lambda: FashionMNISTCNN(hidden_layer_dimension=config["hidden_dim"], output_dimension=config["output_dim"]),
}
model = model_map.get(model_name, model_map["cnn"])().to(device)

loss_map = {
    "cross_entropy": nn.CrossEntropyLoss(),
}
loss_func = loss_map.get(config["loss_func"], nn.CrossEntropyLoss())

optimizer_map = {
    "sgd": lambda: optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["optim_momentum"]),
    "adam": lambda: optim.Adam(model.parameters(), lr=config["learning_rate"]),
    "adamW": lambda: optim.AdamW(model.parameters(), lr=config["learning_rate"]),
}
optimizer = optimizer_map.get(config["optimizer"], optimizer_map["adam"])()

train_model(model, train_loader, loss_func, optimizer, device, epochs=epochs)
test_loss, test_acc = evaluate(model, test_loader, loss_func, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

save_dir = Path("models")
save_dir.mkdir(exist_ok=True)

save_path = save_dir / f"{model_name}_fashionmnist_epoch{epochs}.pth"
torch.save(model.state_dict(), save_path)
