import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

class FocusDataset(Dataset):
    def __init__(self, file_list, transform=None, class_to_idx=None):
        self.file_list = file_list
        self.transform = transform

        # 自动收集所有类别前缀，并创建类别到ID的映射
        if class_to_idx is None:
            self.class_names = self._extract_class_names()
            self.class_to_idx = {name: idx for idx, name in enumerate(sorted(self.class_names))}
        else:
            self.class_to_idx = class_to_idx

    def _extract_class_names(self):
        class_names = set()
        for path in self.file_list:
            class_name = extract_class_name_from_path(path)
            class_names.add(class_name)
        return class_names

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        class_name = extract_class_name_from_path(img_path)
        label = self.class_to_idx[class_name]

        return img_transformed, label

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    return train_transforms, test_transforms