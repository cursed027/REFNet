import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class RefinerDataset(Dataset):
    def __init__(self, degraded_dir, histo_dir, gt_dir, crop_size=256):
        self.degraded_dir = degraded_dir
        self.histo_dir = histo_dir
        self.gt_dir = gt_dir
        self.crop_size = crop_size
        
        # Assume identical filenames across all three directories
        self.image_names = sorted(os.listdir(degraded_dir))

    def _load_image_cv2(self, path):
        # cv2 reads extremely fast in BGR. We convert to RGB, normalize, and permute to [C, H, W]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        return img_tensor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        I_d = self._load_image_cv2(os.path.join(self.degraded_dir, img_name))
        I_h = self._load_image_cv2(os.path.join(self.histo_dir, img_name))
        I_gt = self._load_image_cv2(os.path.join(self.gt_dir, img_name))

        # --- AUGMENTATIONS ---
        # 1. Random Crop (Using pure PyTorch slicing for speed)
        if self.crop_size and I_d.shape[1] >= self.crop_size and I_d.shape[2] >= self.crop_size:
            h_start = random.randint(0, I_d.shape[1] - self.crop_size)
            w_start = random.randint(0, I_d.shape[2] - self.crop_size)
            
            I_d = I_d[:, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size]
            I_h = I_h[:, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size]
            I_gt = I_gt[:, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size]

        # 2. Random Horizontal Flip (Safe Rain Physics)
        if random.random() > 0.5:
            I_d = torch.flip(I_d, [2])
            I_h = torch.flip(I_h, [2])
            I_gt = torch.flip(I_gt, [2])

        # --- DIFFERENCE CHANNEL HACK ---
        diff = I_d - I_h
        input_9ch = torch.cat([I_d, I_h, diff], dim=0)
        
        return input_9ch, I_gt
