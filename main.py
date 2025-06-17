import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from PIL import Image

# Directories
OLD_IMAGE_DIR = r"C:\Users\User\Notebook\project\SECOND_train_set\im1"
NEW_IMAGE_DIR = r"C:\Users\User\Notebook\project\SECOND_train_set\im2"
OLD_LABEL_DIR = r"C:\Users\User\Notebook\project\SECOND_train_set\label1"
NEW_LABEL_DIR = r"C:\Users\User\Notebook\project\SECOND_train_set\label2"

# Hyperparameters
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 10
MODEL_PATH = "scd_model.pth"
NUM_CLASSES = 7  # 6 change classes + no change


# Dataset Definition
class SECOND_Dataset(Dataset):
    def __init__(self, image_dir1, image_dir2, label_dir1, label_dir2, transform=None):
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2
        self.label_dir1 = label_dir1
        self.label_dir2 = label_dir2
        self.transform = transform
        self.image_names = os.listdir(image_dir1)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img1_path = os.path.join(self.image_dir1, img_name)
        img2_path = os.path.join(self.image_dir2, img_name)
        label_path = os.path.join(self.label_dir2, img_name)

        # Load images and labels
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Grayscale for segmentation masks

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            label = self.transform(label)

        # Remove channel dimension from label (squeeze to [H, W])
        label = label.squeeze(0)  # Explicitly squeeze the channel dimension

        return torch.cat([img1, img2], dim=0), label


# Transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# DataLoader
dataset = SECOND_Dataset(OLD_IMAGE_DIR, NEW_IMAGE_DIR, OLD_LABEL_DIR, NEW_LABEL_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model Definition with Fixes Applied
class ChangeDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ChangeDetectionModel, self).__init__()
        # Load DeepLabV3 with ResNet50 backbone and updated weights API
        self.model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # Modify input layer to accept 6 channels instead of 3
        self.model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Update classifier for the number of classes (including background)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']


# Initialize Model and Training Components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChangeDetectionModel(NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.long().squeeze(1).to(device)  # Ensure correct shape

            optimizer.zero_grad()
            outputs = model(images)  # Model output: [batch_size, num_classes, height, width]
            loss = criterion(outputs, labels)  # Labels: [batch_size, height, width]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")


if __name__ == "__main__":
    train()
