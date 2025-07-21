import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.metrics import f1_score
import logging

# ========== ЛОГИРОВАНИЕ ==========
logging.basicConfig(
    filename='./logs/training_resnet18.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# ========== ТРАНСФОРМАЦИИ ==========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ResNet требует 3 канала
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========== ДАТАСЕТЫ ==========
train_dataset = datasets.ImageFolder('./data/train', transform=transform)
val_dataset = datasets.ImageFolder('./data/val', transform=transform)
test_dataset = datasets.ImageFolder('./data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========== МОДЕЛЬ RESNET ==========
class ResNetBinary(nn.Module):
    def __init__(self):
        super(ResNetBinary, self).__init__()
        self.base_model = resnet18(pretrained=True)

        # Если хочешь использовать оригинальный 1-канальный вход (вместо Grayscale -> 3)
        # self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Заменим последний слой
        self.base_model.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # для BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.base_model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetBinary().to(device)

# ========== ВЕС ДЛЯ КЛАССОВ ==========
weight = 1341 / 3875
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ========== ОБУЧЕНИЕ ==========
epochs = 10
patience = 3
best_f1 = 0.0
no_improvement = 0

logger.info("Начало обучения ResNet18...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            percent = 100 * (batch_idx + 1) / len(train_loader)
            logger.info(f"Эпоха {epoch+1}/{epochs} — {percent:.1f}% — loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    logger.info(f"Эпоха [{epoch+1}/{epochs}], loss: {epoch_loss:.4f}")

    # ====== ОЦЕНКА НА ВАЛИДАЦИИ ======
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    f1 = f1_score(val_labels, val_preds)
    logger.info(f"F1 на валидации после эпохи {epoch+1}: {f1:.4f}")

    # ====== РАННЯЯ ОСТАНОВКА ======
    if f1 > best_f1:
        torch.save(model.state_dict(), './models/resnet18_binary.pth')
        best_f1 = f1
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= patience:
        logger.info(f"Раннее завершение: F1 не улучшалась {patience} эпох.")
        break

logger.info("Обучение завершено.")
