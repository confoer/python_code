import os
import torch
import numpy as np
import config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_dataloaders():
    train_transform, val_transform = get_transforms()
    
    full_dataset = EmotionDataset(os.path.join(config.config.data_root, "train"), transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 修正验证集的transform
    val_dataset.dataset.transform = val_transform
    
    test_dataset = EmotionDataset(os.path.join(config.config.data_root, "test"), transform=val_transform)
    
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=config.config.batch_size, 
                          shuffle=True, num_workers=config.config.num_workers),
        "val": DataLoader(val_dataset, batch_size=config.config.batch_size,
                        shuffle=False, num_workers=config.config.num_workers),
        "test": DataLoader(test_dataset, batch_size=config.config.batch_size,
                         shuffle=False, num_workers=config.config.num_workers)
    }
    
    return dataloaders