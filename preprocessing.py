import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image

num_classes = 2
learning_rate = 0.001
dropout_rate = 0.5
batch_size = 32
num_epochs = 10

class ResNetIrisTumor(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(ResNetIrisTumor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

model = ResNetIrisTumor(num_classes=num_classes, dropout_rate=dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

class CustomDataset(Dataset):
    def __init__(self, folder_path, label, transform=None):
        self.folder_path = folder_path
        self.label = label
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        filtered_image = cv2.medianBlur(grayscale_image, 5)
        
        edges = cv2.Canny(filtered_image, threshold1=50, threshold2=150)
        
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(edges_3channel)
        
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)
        
        return image, self.label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

tumor_path = "/content/drive/MyDrive/INFOSYS SPRINGBOARD/IRIS/YES"
no_tumor_path = "/content/drive/MyDrive/INFOSYS SPRINGBOARD/IRIS/NO"

tumor_dataset = CustomDataset(tumor_path, label=1, transform=transform)
no_tumor_dataset = CustomDataset(no_tumor_path, label=0, transform=transform)

combined_dataset = ConcatDataset([tumor_dataset, no_tumor_dataset])
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_data, val_data = random_split(combined_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_epoch_acc = 100. * val_correct / val_total
    print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%')
