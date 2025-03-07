import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_transform: transforms.Compose = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(256),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform: transforms.Compose = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])  

class FlashbangModel(nn.Module):
    def __init__(self):
        super(FlashbangModel, self).__init__()
        
        self.model = models.resnet18(weights=None)
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    
    start_time = time.time()   
    print("Training started")
    
    for epoch in tqdm(range(epochs)):
        loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        total_loss = loss / len(train_loader)
        accuracy = correct / total * 100
    
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f} - Accuracy: {accuracy:.2f}")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        accuracy = correct / total * 100
    
        print(f"Test Accuracy: {accuracy:.2f}")
        
def main():
    parser = argparse.ArgumentParser(description="Train a model to detect flashbangs")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--model-path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    
    IMAGES_DIR = pathlib.Path(args.images_dir)
    MODEL_PATH = pathlib.Path(args.model_path)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    CPU_COUNT = os.cpu_count()
    
    print(f"Using {device} for training")
    print(f"Training on {IMAGES_DIR} with {EPOCHS} epochs and batch size of {BATCH_SIZE}")
    print(f"Using {CPU_COUNT} CPU cores for data loading")

    dataset = datasets.ImageFolder(root=IMAGES_DIR, transform=training_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size  

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_dataset.dataset.transform = test_transform

    train_labels = [dataset.targets[i] for i in train_dataset.indices]

    class_counts = np.bincount(train_labels)
    class_weights = [1.0 / count for count in class_counts]

    sample_weights = [class_weights[label] for label in train_labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT, shuffle=False)

    model = FlashbangModel().to(device)

    class_weights = torch.tensor([1.0, 20.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, loss_fn, optimizer, device, epochs=EPOCHS)
    evaluate_model(model, test_loader, loss_fn, device)
    
    if not os.path.exists(MODEL_PATH.parent):
        os.makedirs(MODEL_PATH.parent)
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
if __name__ == "__main__":
    main()