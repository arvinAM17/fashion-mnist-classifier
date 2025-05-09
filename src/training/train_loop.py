import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str, 
    epoch: int = 0,
    writer: SummaryWriter = None
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    i = 0

    for images, labels in data_loader:
        i += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 100 == 0:
            train_acc = accuracy = 100 * correct / total
            if writer:
                writer.add_scalar("Loss/training", loss.item(), (i + epoch * len(data_loader)) // 100)
                writer.add_scalar("Accuracy/training", train_acc, (i + epoch * len(data_loader)) // 100)
        

    accuracy = 100 * correct / total
    return total_loss / len(data_loader), accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_func: nn.Module,
    device: str
):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(data_loader), accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 10,
    writer: SummaryWriter = None
):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_func, optimizer, device, epoch, writer)
        val_loss, val_acc = evaluate(model, validation_loader, loss_func, device)

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
