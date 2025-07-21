import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, confusion_matrix, fbeta_score
from PIL import Image
from pathlib import Path
import logging

# ——— Logging setup ———
logging.basicConfig(
    filename='./logs/training_densenet161.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# ——— Hyperparameters & globals ———
EPOCHS = 10
PATIENCE = 2
BEST_METRIC = 0.0  # precision при условии recall>=MIN_RECALL
NO_IMPROV = 0
RESCALE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = Path('./data/train')
VAL_DIR = Path('./data/val')
TEST_DIR = Path('./data/test')
DATA_MODES = ['train', 'val', 'test']
MIN_RECALL = 0.87  # минимальная требуемая чувствительность
THRESHOLD = 0.5  # стартовый порог
FREEZE_EPOCHS = 3  # число эпох, в которые backbone заморожен


# ——— Dataset ———
class CancerDataset(Dataset):
    def __init__(self, files, mode):
        self.files = sorted(files)
        self.mode = mode
        self.labels = np.array([1 if p.parent.name == 'Cancer' else 0 for p in self.files])
        if mode not in DATA_MODES:
            raise ValueError(f"mode must be one of {DATA_MODES}")

        pil_transforms = [
            transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
        ]
        if mode == 'train':
            pil_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.1, 0.1, 0.1),
            ]

        tensor_transforms = [
            transforms.ToTensor(),
            transforms.Normalize([0.4875] * 3, [0.2456] * 3),
        ]
        post_tensor = []
        if mode == 'train':
            post_tensor += [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ]

        self.transform = transforms.Compose(pil_transforms + tensor_transforms + post_tensor)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        x = self.transform(img)
        if self.mode == 'test':
            return x
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ——— Threshold search: максимизируем precision при recall>=MIN_RECALL ———
def find_threshold_at_min_recall(y_true, probs, min_recall=MIN_RECALL):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # thresholds.shape == (len(precision)-1,)
    valid = recall[:-1] >= min_recall
    if valid.any():
        prec_valid = precision[:-1][valid]
        th_valid = thresholds[valid]
        idx = np.argmax(prec_valid)
        return float(th_valid[idx])
    # fallback на макс F1
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    # возьмём порог, дающий max f1, но убедимся в границах thresholds
    idx = np.nanargmax(f1s[:-1])
    return float(thresholds[idx])


# ——— Training & Evaluation ———
def fit_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > THRESHOLD).astype(int)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

        total_loss += loss.item() * x.size(0)

    loss_avg = total_loss / len(loader.dataset)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    logger.info(f"Train loss {loss_avg:.4f} | P {precision:.4f} R {recall:.4f} S {specificity:.4f}")
    return loss_avg, precision, recall, specificity


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
            total_loss += loss.item() * x.size(0)

    loss_avg = total_loss / len(loader.dataset)
    best_t = find_threshold_at_min_recall(all_labels, all_probs)
    preds = (np.array(all_probs) > best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    logger.info(f"Val loss {loss_avg:.4f} @thr={best_t:.3f} | P {precision:.4f} R {recall:.4f} S {specificity:.4f}")
    return loss_avg, precision, recall, specificity, best_t


def train(train_ds, val_ds, model, optimizer, epochs, batch_size):
    global BEST_METRIC, NO_IMPROV, THRESHOLD
    print("--------------------------------")
    print("len(train_ds) =", len(train_ds))
    print("len(val_ds)   =", len(val_ds))
    print("--------------------------------")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # вычисляем pos_weight для дисбаланса
    counts = np.bincount(train_ds.labels)
    pos = counts[1] if len(counts) > 1 else 1
    neg = counts[0]
    pos_weight = torch.tensor((neg / pos)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # изначально замораживаем backbone
    for p in model.features.parameters():
        p.requires_grad = False

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{epochs} | LR: {lr:.6f}")

        # градационная разморозка
        if epoch == FREEZE_EPOCHS + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            logger.info(f"Unfroze backbone at epoch {epoch}")

        tr_loss, tr_p, tr_r, tr_s = fit_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_p, val_r, val_s, best_t = eval_epoch(model, val_loader, criterion)

        THRESHOLD = best_t  # сохраняем лучшую модель по precision при достаточной recall
        if val_p > BEST_METRIC and val_r >= MIN_RECALL:
            BEST_METRIC = val_p
            NO_IMPROV = 0
            torch.save(model.state_dict(), './models/densenet161_besttry.pth')

            with open('./models/threshold.txt', 'w') as f:
                f.write(str(THRESHOLD))
            logger.info("Saved new best model.")
        elif val_r >= MIN_RECALL:
            NO_IMPROV += 1

        print(f"[{epoch}/{epochs}] Val P: {val_p:.4f} R: {val_r:.4f} S: {val_s:.4f} @thr={best_t:.3f}")

        scheduler.step(val_p)
        if NO_IMPROV >= PATIENCE:
            logger.info(f"Early stopping after {PATIENCE} epochs without improvement.")
            break


def predict(model, test_loader):
    model.eval()
    out = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(DEVICE)
            logits = model(x).squeeze()
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


# ——— Main ———
if __name__ == '__main__':
    torch.manual_seed(42)  #
    np.random.seed(42)  # 4

    train_files = list(TRAIN_DIR.rglob('*.jpeg'))
    val_files = list(VAL_DIR.rglob('*.jpeg'))
    test_files = list(TEST_DIR.rglob('*.jpeg'))
    print(len(train_files), len(val_files), len(test_files))

    train_ds = CancerDataset(train_files, mode='train')
    val_ds = CancerDataset(val_files, mode='val')
    test_ds = CancerDataset(test_files, mode='test')

    model = models.densenet161(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier.in_features, 1)
    )
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 5e-7, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    ])

    logger.info(f"Starting training on {DEVICE}")
    train(train_ds, val_ds, model, optimizer, EPOCHS, batch_size=16)

    logger.info("Training complete, model and threshold saved.")
