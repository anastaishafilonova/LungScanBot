import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import numpy as np
import logging

# ========== ЛОГИРОВАНИЕ ==========
logging.basicConfig(
    filename='./logs/model.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# ========== ПРЕОБРАЗОВАНИЯ ==========
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x)),
    transforms.Normalize([0.5], [0.5])
])

transform_val_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ========== ЗАГРУЗКА ДАННЫХ ==========
train_dataset = datasets.ImageFolder('./data/train', transform=transform_val_test)
val_dataset = datasets.ImageFolder('./data/val', transform=transform_val_test)
test_dataset = datasets.ImageFolder('./data/test', transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ========== МОДЕЛЬ ==========
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.fc_block(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# ========== ЛОСС И ОПТИМАЙЗЕР ==========
base_criterion = nn.BCEWithLogitsLoss()
alpha = 0.05  # вес штрафа за неуверенность
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ========== ОБУЧЕНИЕ ==========
epochs = 20
patience = 3
best_f1 = 0.0
no_improvement = 0

logger.info("Начало обучения...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)

        # BCE + штраф за логиты ближе к нулю
        bce_loss = base_criterion(outputs, labels)
        confidence_penalty = alpha * torch.mean(torch.exp(-torch.abs(outputs)))
        loss = bce_loss + confidence_penalty

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    logger.info(f"Эпоха [{epoch + 1}/{epochs}], loss: {epoch_loss:.4f}")

    # === ВАЛИДАЦИЯ ===
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    f1 = f1_score(val_labels, val_preds)
    logger.info(f"F1 на валидации после эпохи {epoch + 1}: {f1:.4f}")

    if f1 > best_f1 + 0.01:
        best_f1 = f1
        no_improvement = 0
        torch.save(model.state_dict(), './models/model.pth')
        logger.info("F1 улучшилось, модель сохранена.")
    else:
        no_improvement += 1
        logger.info(f"Нет улучшения. Счётчик ранней остановки: {no_improvement}/{patience}")

    if no_improvement >= patience:
        logger.info("Ранняя остановка.")
        break

logger.info("Обучение завершено.")
