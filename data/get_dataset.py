from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import os
from PIL import Image
import sys
import numpy as np
import torch

image_path = r"F:\New\Coding\GAF_VIT\images"
image_keys = [
  "XQ-11",
  "XQ-12",
  "XQ-14",
  "XQ-15",
  "XQ-16",
  "XQ-17",
  "XQ-18"
]
def get_images_path(image_dir, image_key):
  # if os.path.exists(image_path):
    return None

transform = transforms.Compose([
  transforms.Resize((128,128)),
  transforms.ToTensor(),
])

class BatteryDataset(Dataset):
  def __init__(self,img_dir=image_path,transform=None):
    self.data = None
    self.img_dir = image_path
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    img_name = os.path.join(self.img_dif, self.data.ioloc)
    image = Image.open(img_name)

    label = float(self.data.iloc)

    if self.transform:
      image = self.transform(image)

    return image, label
