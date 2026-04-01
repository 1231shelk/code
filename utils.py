import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def extract_class_name_from_path(path):
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    if len(name_without_ext) < 5:
        raise ValueError(f"文件名太短: {name_without_ext}")
    return name_without_ext[:-5]


# ======================= 保存/加载函数 =======================
def save_checkpoint(epoch, model, optimizer, scheduler, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch']
    print(f"成功加载 checkpoint (epoch={start_epoch})")
    return start_epoch


