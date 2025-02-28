import os
import pathlib

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
import data.get_soh as gs

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
train_image_keys = [
  "XQ-11",
  "XQ-12",
  "XQ-14",
  "XQ-15"
]
test_image_keys = [
  "XQ-16",
  "XQ-17"
]
val_image_keys = ["XQ-18"]

# 定义数据增强


def get_images_path(image_dir: str, image_keys: []):
  if not os.path.exists(image_dir):
    print("the image_dir is not exist,please input a new path")
  images = []
  for image_key in image_keys:
    path = pathlib.Path(os.path.join(image_path, image_key + "-images"))
    for P in path.iterdir():
      images.append(P.__str__())

  return images

# def get_images_path_data(image_dir:str, image_keys:[]):
#   images = []
#   for image_key in image_keys:
#     images.append(get_images_path(image_dir, image_key))
#   return images

def get_labels(image_paths: [], SOH_Labels: dict):
  labels = []
  for path in image_paths:
    key = "XQ-" + path.split("\\")[-1].split("-")[1]
    idx = path.split("\\")[-1].split("-")[-1].split(".")[0]
    label = SOH_Labels[key][int(idx)]
    labels.append(label)
  return labels


# paths = get_images_path(image_path, image_keys)
# labels = get_labels(paths, gs.get_soh_labels(image_keys))


# print(labels)
# key = "XQ-" + paths[0].split("\\")[-1].split("-")[1]
# print(key)
# idx = paths[0].split("\\")[-1].split("-")[-1].split(".")[0]
# print(idx)
# print(labels[key][int(idx)])


def get_transform():
  transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
  ])
  return transform


class BatteryDataset(Dataset):
  def __init__(self, img_dir=image_path, img_keys=image_keys,labels = [], transform=None):
    self.img_dir = img_dir
    self.img_keys = img_keys
    self.transform = transform
    self.path_data = get_images_path(img_dir, img_keys)
    # self.labels = gs.get_soh_labels()
    self.labels = labels

  def __len__(self):
    return len(self.path_data)

  def __getitem__(self, index):
    img_path = self.path_data[index]
    # img_path_key = "XQ-" + image_path.split("\\")[-1].split("-")[1]
    # img_path_index = image_path.split("\\")[-1].split("-")[-1].split(".")[0]
    # label = self.labels[img_path_key][int(img_path_index)]
    label = self.labels[index]
    image = Image.open(img_path).convert("RGB")
    label = torch.tensor(label, dtype=torch.float)

    if self.transform:
      image = self.transform(image)

    return image, label
def get_train_transform():
  return transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # 示例增强
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准参数
                         std=[0.229, 0.224, 0.225])
  ])


def get_val_transform():
  return transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])


# 创建完整数据集
def create_loaders(batch_size=32):
  # 获取所有路径和标签
  all_paths = get_images_path(image_path, image_keys)
  all_labels = get_labels(all_paths, gs.get_soh_labels(image_keys))  # 假设gs已定义

  # 按你的划分策略分离数据
  train_paths = get_images_path(image_path, train_image_keys)
  val_paths = get_images_path(image_path, val_image_keys)
  test_paths = get_images_path(image_path, test_image_keys)

  # 获取对应的标签切片
  def get_subset_labels(full_paths, subset_paths):
    idxs = [full_paths.index(p) for p in subset_paths]
    return [all_labels[i] for i in idxs]

  train_labels = get_subset_labels(all_paths, train_paths)
  val_labels = get_subset_labels(all_paths, val_paths)
  test_labels = get_subset_labels(all_paths, test_paths)

  # 创建数据集实例
  train_dataset = BatteryDataset(
    img_keys=train_image_keys,
    labels=train_labels,
    transform=get_train_transform()
  )

  val_dataset = BatteryDataset(
    img_keys=val_image_keys,
    labels=val_labels,
    transform=get_val_transform()
  )

  test_dataset = BatteryDataset(
    img_keys=test_image_keys,
    labels=test_labels,
    transform=get_val_transform()
  )

  # 创建DataLoader
  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
  )

  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
  )

  return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
  train_loader, val_loader, test_loader = create_loaders(batch_size=32)

  # 验证数据流
  for images, labels in train_loader:
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")
    break

  for images, labels in val_loader:
    print(f"Val Batch - Images: {images.shape}, Labels: {labels.shape}")
    break