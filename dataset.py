import os
import cv2
import torch
import pandas as pd 
from torch.utils.data import Dataset


class brain_mri_dataset(Dataset):
    
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data.iloc[index]["Path"]
        image_name = self.data.iloc[index]["Name"]
        label      = self.data.iloc[index]["Label"]
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.resize(image, (224, 224))
        image = image / 255.0 
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)
        
        return {
            "mri": image,
            "name": image_name,
            "target": label
        }