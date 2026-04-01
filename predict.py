import glob
import torch
import config as cfg
from torch.utils.data import DataLoader
# from dataset import FocusDataset, get_transforms
from muti_dataset import FocusDataset, get_transforms
from model import get_vit_model, ConvStemViT, CVT
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 模型路径(可改成最佳模型)
model_path = f"{cfg.MODEL_SAVE_DIR}/87.pt"

# 测试数据
test_list = glob.glob(cfg.TEST_DIR + '/*.tif')
_, test_tf = get_transforms()
test_dataset = FocusDataset(test_list, test_tf)

# 提取类别名(一定要用 Dataset 对象,不是 DataLoader)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(idx_to_class)

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

# 加载模型
model = ConvStemViT(image_size=224, num_classes=num_classes, device=cfg.DEVICE)
checkpoint = torch.load(model_path, map_location=cfg.DEVICE)

# 如果保存的是完整checkpoint 或者是纯 state_dict
if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.to(cfg.DEVICE)
model.eval()

# 预测
all_preds = []
all_labels = []

with torch.no_grad():
    for data, label in test_loader:
        data = data.to(cfg.DEVICE)
        output = model(data)
        preds = output.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(label.numpy())

# 输出分类报告
print(classification_report(all_labels, all_preds, target_names=target_names))

# ======================= 混淆矩阵 =======================
cm = confusion_matrix(all_labels, all_preds)

# 保存混淆矩阵数据为 CSV
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
os.makedirs(cfg.CSV_SAVE_DIR, exist_ok=True)
cm_csv_path = os.path.join(cfg.CSV_SAVE_DIR, "E87amp_confusion_matrix.csv")
cm_df.to_csv(cm_csv_path, index=True)
print(f"混淆矩阵已保存到: {cm_csv_path}")

# 可视化混淆矩阵(并保存为图片)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_img_path = os.path.join(cfg.CSV_SAVE_DIR, "E87amp_confusion_matrix.png")
plt.savefig(cm_img_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"混淆矩阵图片已保存到: {cm_img_path}")




