<<<<<<< HEAD
import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

train_dir = r"C:\Users\Public\Documents\VS_Projects\Web_App\flowers_split\train"
val_dir = r"C:\Users\Public\Documents\VS_Projects\Web_App\flowers_split\val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"✅ Epoch {epoch+1} complete. Loss: {running_loss:.4f}")

torch.save(model.state_dict(), r"C:\Users\Public\Documents\VS_Projects\Web_App\model_state_dict.pth")
print("✅ Model weights saved as model_state_dict.pth")
=======
import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

train_dir = r"C:\Users\Public\Documents\VS_Projects\Web_App\flowers_split\train"
val_dir = r"C:\Users\Public\Documents\VS_Projects\Web_App\flowers_split\val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"✅ Epoch {epoch+1} complete. Loss: {running_loss:.4f}")

torch.save(model.state_dict(), r"C:\Users\Public\Documents\VS_Projects\Web_App\model_state_dict.pth")
print("✅ Model weights saved as model_state_dict.pth")
>>>>>>> 276e653 (Initial commit)
