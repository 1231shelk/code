import os
import glob
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import config as cfg
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model import get_vit_model, ConvStemViT, CVT
# from dataset import FocusDataset, get_transforms
from muti_dataset import FocusDataset, get_transforms
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ======================= 配置 =======================
device = cfg.DEVICE
seed_everything(cfg.SEED)
print(f"已固定随机种子: {cfg.SEED}")

# 创建保存目录
os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(cfg.MATRIX_SAVE_DIR, exist_ok=True)
os.makedirs(cfg.CSV_SAVE_DIR, exist_ok=True)


# ======================= 数据准备 =======================
train_list = glob.glob(os.path.join(cfg.TRAIN_DIR, '*.tif'))
valid_list = glob.glob(os.path.join(cfg.VALID_DIR, '*.tif'))  # 采用手动划分
# train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=cfg.SEED)
test_list = glob.glob(os.path.join(cfg.TEST_DIR, '*.tif'))
labels = [extract_class_name_from_path(p) for p in train_list]

# print(f"前10个标签示例: {labels[:10]}")
print(f"labels数量: {len(labels)}")
print(f"训练集: {len(train_list)}, 验证集: {len(valid_list)}, 测试集: {len(test_list)}")

train_tf, test_tf = get_transforms()
train_data = FocusDataset(train_list, transform=train_tf)
valid_data = FocusDataset(valid_list, transform=test_tf, class_to_idx=train_data.class_to_idx)
num_classes = len(train_data.class_to_idx)

# train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
valid_loader = DataLoader(valid_data, batch_size=cfg.BATCH_SIZE, shuffle=False,num_workers=0, pin_memory=True, persistent_workers=False)

print(f"分类数量: {num_classes}")
print("类别及标签编号:")
for class_name, label in train_data.class_to_idx.items():
    print(f"{class_name}: {label}")


# ======================= 模型/损失/优化器 =======================
# model = get_vit_model((image_size=224, num_classes=num_classes, device=device)
# model = CVT(image_size=224, num_classes=num_classes, device=device)
model = ConvStemViT(image_size=224, num_classes=num_classes, device=device)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
scheduler = StepLR(optimizer, step_size=1, gamma=cfg.GAMMA)


# ======================= 训练循环 =======================
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# ---------- 训练准备 ----------
start_epoch = 0
resume_path = None  # 如果要续训，这里填上保存的 ckpt 路径，例如 "checkpoints/10.pt"

if resume_path and os.path.exists(resume_path):
    start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler, device)

for epoch in range(start_epoch, cfg.EPOCHS):
    # ---------- 训练 ----------
    model.train()
    epoch_loss, epoch_correct = 0, 0

    for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_correct += (output.argmax(1) == label).sum().item()

    train_losses.append(epoch_loss / len(train_loader))
    train_accuracies.append(epoch_correct / len(train_data))

    # ---------- 验证 ----------
    model.eval()
    val_loss, val_correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)

            val_loss += loss.item()
            val_correct += (output.argmax(1) == label).sum().item()

            all_preds.extend(output.argmax(1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    val_losses.append(val_loss / len(valid_loader))
    val_accuracies.append(val_correct / len(valid_data))

    # ---------- 学习率调度 ----------
    scheduler.step()

    # ---------- 保存模型 ----------
    ckpt_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{epoch + 1}.pt")
    save_checkpoint(epoch + 1, model, optimizer, scheduler, ckpt_path)

    print(f"Epoch {epoch + 1}: Train Loss={train_losses[-1]:.4f}, Train Acc={train_accuracies[-1]:.4f}, "
          f"Val Loss={val_losses[-1]:.4f}, Val Acc={val_accuracies[-1]:.4f}")

    # ---------- 混淆矩阵 ----------
    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(cfg.MATRIX_SAVE_DIR, f'confusion_matrix_epoch_{epoch+1}.npy'), cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(train_data.class_to_idx.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
    plt.savefig(os.path.join(cfg.MATRIX_SAVE_DIR, f'confusion_matrix_epoch_{epoch+1}.png'))
    plt.close()

    # ======================= 保存训练历史（逐轮追加） =======================
    history_row = pd.DataFrame([{
        'epoch': epoch + 1,
        'train_loss': train_losses[-1],
        'train_acc': train_accuracies[-1],
        'val_loss': val_losses[-1],
        'val_acc': val_accuracies[-1]
    }])

    csv_path = os.path.join(cfg.CSV_SAVE_DIR, "training_history.csv")

    # 如果文件不存在，写入表头；否则追加不写表头
    if not os.path.exists(csv_path):
        history_row.to_csv(csv_path, index=False, mode='w')
    else:
        history_row.to_csv(csv_path, index=False, mode='a', header=False)


# ======================= 保存最终训练历史 =======================
history_df = pd.DataFrame({
    'epoch': list(range(1, cfg.EPOCHS+1)),
    'train_loss': train_losses,
    'train_acc': train_accuracies,
    'val_loss': val_losses,
    'val_acc': val_accuracies
})
history_df.to_csv(os.path.join(cfg.CSV_SAVE_DIR, "training_history02.csv"), index=False)

# ======================= 绘制训练曲线 =======================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_df['train_loss'], label='Train Loss')
plt.plot(history_df['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history_df['train_acc'], label='Train Acc')
plt.plot(history_df['val_acc'], label='Val Acc')
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig(os.path.join(cfg.CSV_SAVE_DIR, "training_curve.png"))
plt.close()


