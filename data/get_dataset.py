import os
import pathlib
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
import data.get_feature as gf
from data.xjbattery import Battery

image_path = r"D:\1New\Coding\GAF_VIT\images3"
xj_image_path = r"D:\1New\Coding\GAF_VIT\xj_images"
tj_image_path = r"D:\1New\Coding\GAF_VIT\tj_images\Dataset_1_NCA_battery"
image_keys = ["XQ-11", "XQ-12", "XQ-14", "XQ-15", "XQ-16", "XQ-17", "XQ-18"]
train_image_keys = ["XQ-11", "XQ-12", "XQ-14", "XQ-15"]
test_image_keys = ["XQ-16", "XQ-17"]
val_image_keys = ["XQ-18"]
xj_image_keys = [
  "2C_battery-1",
  "2C_battery-2",
  "2C_battery-3",
  "2C_battery-4",
  "2C_battery-5",
  "2C_battery-6",
  "2C_battery-7",
  "2C_battery-8",
]
xj_train_image_keys = ["2C_battery-1", "2C_battery-2", "2C_battery-3", "2C_battery-4"]
xj_test_image_keys = [
  "2C_battery-5",
  "2C_battery-6",
  "2C_battery-7",
]
xj_val_image_keys = ["2C_battery-8"]

# Dataset_1_NCA_battery
tj_image_keys = {
  "CY25-025_1-#": ["CY25-025_1-#" + str(i) for i in range(1, 8)],
  "CY25-1_1-#": ["CY25-1_1-#" + str(i) for i in range(1, 10)],
  "CY25-05_1-#": ["CY25-05_1-#" + str(i) for i in range(1, 20)],
  "CY35-05_1-#": ["CY35-05_1-#" + str(i) for i in range(1, 4)],
  "CY45-05_1-#": ["CY45-05_1-#" + str(i) for i in range(1, 29)],
}

tj_train_image_keys = (
  tj_image_keys["CY25-025_1-#"][0:5]
  + tj_image_keys["CY25-1_1-#"][0:6]
  + tj_image_keys["CY25-05_1-#"][0:14]
  + tj_image_keys["CY35-05_1-#"][0:1]
  + tj_image_keys["CY45-05_1-#"][0:21]
)
tj_val_image_keys = (
  tj_image_keys["CY25-025_1-#"][5:6]
  + tj_image_keys["CY25-1_1-#"][6:8]
  + tj_image_keys["CY25-05_1-#"][14:17]
  + tj_image_keys["CY35-05_1-#"][1:2]
  + tj_image_keys["CY45-05_1-#"][21:25]
)
tj_test_image_keys = (
  tj_image_keys["CY25-025_1-#"][6:]
  + tj_image_keys["CY25-1_1-#"][8:]
  + tj_image_keys["CY25-05_1-#"][17:]
  + tj_image_keys["CY35-05_1-#"][2:]
  + tj_image_keys["CY45-05_1-#"][25:]
)
tj_image_keys = [key for keys in list(tj_image_keys.values()) for key in keys]


def get_images_path(image_dir: str, image_keys: [], image_flag: str):
  if not os.path.exists(image_dir):
    print("the image_dir is not exist,please input a new path")
  images = []
  if image_flag == "SE":
    # todo 添加判断xj文件目录
    for image_key in image_keys:
      path = pathlib.Path(os.path.join(image_dir, image_key + "-images"))
      for P in path.iterdir():
        images.append(P.__str__())
  elif image_flag == "XJ":
    for image_key in image_keys:
      path = pathlib.Path(os.path.join(image_dir, image_key))
      path_dir = sorted(path.iterdir(), key=lambda p: p.name)
      for P in path_dir:
        images.append(P.__str__())
  elif image_flag == "TJ":
    # todo 获取TJ图
    for image_key in image_keys:
      path = pathlib.Path(os.path.join(image_dir, image_key))
      for P in path.iterdir():
        images.append(P.__str__())

  return images


def get_labels(image_paths: [], SOH_Labels: dict, label_flag: str):
  labels = []
  if label_flag == "SE":
    for path in image_paths:
      key = "XQ-" + path.split("\\")[-1].split("-")[1]
      idx = path.split("\\")[-1].split("-")[-1].split(".")[0]
      label = SOH_Labels[key][int(idx)]
      labels.append(label)
  elif label_flag == "XJ":
    for path in image_paths:
      key = path.split("\\")[-2]
      idx = int(path.split("\\")[-1].split("-")[-1].split(".")[0]) - 1
      label = SOH_Labels[key][idx]
      labels.append(label)
  elif label_flag == "TJ":
    for path in image_paths:
      key = path.split("\\")[-2]
      idx = int(re.search(r"#(\d+)-\d+\.png", path.split("\\")[-1]).group(1))
      label = SOH_Labels[key][idx]
      labels.append(label)

  return labels


# key = "XQ-" + paths[0].split("\\")[-1].split("-")[1]
# print(key)
# idx = paths[0].split("\\")[-1].split("-")[-1].split(".")[0]
# print(idx)
# print(labels[key][int(idx)])


def get_transform():
  transform = transforms.Compose(
    [
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
    ]
  )
  return transform


class BatteryDataset(Dataset):
  def __init__(
    self, img_dir=image_path, img_keys=None, path_data=None, labels=None, transform=None
  ):
    if img_keys is None:
      img_keys = image_keys
    if labels is None:
      labels = []
    if path_data is None:
      path_data = []
    self.img_dir = img_dir
    self.img_keys = img_keys
    self.transform = transform
    self.path_data = path_data
    # self.labels = gs.get_soh_labels()
    self.labels = labels

  def __len__(self):
    return len(self.path_data)

  def __getitem__(self, index):
    img_path = self.path_data[index]
    label = self.labels[index]
    image = Image.open(img_path).convert("L")
    label = torch.tensor(label, dtype=torch.float)

    if self.transform:
      image = self.transform(image)

    return image, label


def get_train_transform():
  return transforms.Compose(
    [
      transforms.Resize((128, 128)),
      transforms.RandomHorizontalFlip(),  # 示例增强
      transforms.RandomRotation(10),
      transforms.ToTensor(),
      # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准参数
      #                      std=[0.229, 0.224, 0.225])
      transforms.Normalize(mean=0.45, std=0.2),
    ]
  )


def get_val_transform():
  return transforms.Compose(
    [
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      # transforms.Normalize(mean=[0.485, 0.456, 0.406],
      #                      std=[0.229, 0.224, 0.225])
      transforms.Normalize(mean=0.45, std=0.2),
    ]
  )


"""
   self data loader creat
"""


# 创建完整数据集
def create_loaders(path, keys, batch_size=32, loader_flag="SE"):
  # 获取所有路径和标签
  all_paths = get_images_path(path, keys, image_flag=loader_flag)
  all_labels = get_labels(
    all_paths, gf.get_soh_labels(keys, loader_flag), label_flag=loader_flag
  )  # 假设gs已定义

  if loader_flag == "SE":
    train_keys, val_keys, test_keys = train_image_keys, val_image_keys, test_image_keys
  elif loader_flag == "XJ":
    train_keys, val_keys, test_keys = (
      xj_train_image_keys,
      xj_val_image_keys,
      xj_test_image_keys,
    )
  else:
    print("the flag error, please input ")
    return {}

  # 按你的划分策略分离数据
  train_paths = get_images_path(path, train_keys, loader_flag)
  val_paths = get_images_path(path, val_keys, loader_flag)
  test_paths = get_images_path(path, test_keys, loader_flag)

  # 获取对应的标签切片
  def get_subset_labels(full_paths, subset_paths):
    idxs = [full_paths.index(p) for p in subset_paths]
    return [all_labels[i] for i in idxs]

  train_labels = get_subset_labels(all_paths, train_paths)
  val_labels = get_subset_labels(all_paths, val_paths)
  test_labels = get_subset_labels(all_paths, test_paths)

  # 创建数据集实例
  train_dataset = BatteryDataset(
    img_dir=path,
    img_keys=train_keys,
    path_data=train_paths,
    labels=train_labels,
    transform=get_train_transform(),
  )

  val_dataset = BatteryDataset(
    img_dir=path,
    img_keys=val_keys,
    path_data=val_paths,
    labels=val_labels,
    transform=get_val_transform(),
  )

  test_dataset = BatteryDataset(
    img_dir=path,
    img_keys=test_keys,
    path_data=test_paths,
    labels=test_labels,
    transform=get_val_transform(),
  )

  # 创建DataLoader
  train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
  )

  val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
  )

  test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
  )

  return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
  print(tj_image_keys)
  print(tj_train_image_keys)
  print(tj_val_image_keys)
  print(tj_test_image_keys)
  # train_loader, val_loader, test_loader = create_loaders(
  #   xj_image_path, xj_image_keys, loader_flag="XJ"
  # )
  # # 验证数据流
  # for images, labels in train_loader:
  #   print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")
  #   break
  #
  # for images, labels in val_loader:
  #   print(f"Val Batch - Images: {images.shape}, Labels: {labels.shape}")
  #   break
  # paths = get_images_path(tj_image_path, xj_image_keys, "XJ")
  # labels = get_labels(
  #   paths,
  #   gf.get_soh_labels(xj_image_keys, "XJ"), "XJ"
  # )
  # print(labels)
