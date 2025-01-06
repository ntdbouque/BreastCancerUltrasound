'''
Các Function thực hiện load data, chia train test, tạo lớp dataset, load dataloader
'''

import os
import glob
import numpy as np
import cv2 
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


def load_dataloader(dataset_path,
                    test_size,
                    input_size,
                    transform,
                    target_transform=None,
                    batch_size=8,
                    num_workers=95):
    train_data, val_data = split_dataset(dataset_path)
    
    train_dataset = MedicalImageDataset(
        dataset=train_data,
        input_size = input_size,
        transform = transform,
        target_transform=None
    )
    
    val_dataset = MedicalImageDataset(
        dataset=val_data,
        input_size=input_size,
        transform=transform,
        target_transform=None
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=95)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=95)
    
    return  train_dataloader, val_dataloader

def split_dataset(dataset_dir):
    benign, malignant, normal = [], [], []
    benign_images = glob.glob(os.path.join(dataset_dir, 'benign', '*.png'))
    malignant_images = glob.glob(os.path.join(dataset_dir, 'malignant', '*.png'))
    normal_images = glob.glob(os.path.join(dataset_dir, 'normal', '*.png'))
    
    for mask in benign_images:
        if "_mask" in mask:
            benign.append((0, mask))
    for mask in malignant_images:
        if "_mask" in mask:
            malignant.append((1, mask))
    for mask in normal_images:
        if "_mask" in mask:
            normal.append((2, mask))
    
    all_data = benign + malignant + normal
    np.random.shuffle(all_data)
    
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    #test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    return train_data, val_data

class MedicalImageDataset(Dataset):
    def __init__(self, dataset, input_size, transform=None, target_transform=None):
        self.dataset = dataset    
        self.input_size = input_size
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        label = int(self.dataset[idx][0])
        mask_path = self.dataset[idx][1]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)    
        
        mask = mask /255

        if self.target_transform:
            mask = self.target_transform(mask)
            
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(dim=0)
        return {'mask': mask, 'label': label} 

