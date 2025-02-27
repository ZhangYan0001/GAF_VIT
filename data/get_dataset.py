import os
import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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


def get_images_path(image_dir: str, image_keys: []):
  if not os.path.exists(image_dir):
    print("the image_dir is not exist,please input a new path")
  # images_dict = {}
  images = []
  for image_key in image_keys:
    path = pathlib.Path(os.path.join(image_path, image_key + "-images"))
    for P in path.iterdir():
      images.append(P.__str__())

  # images_dict[image_key] = images
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


paths = get_images_path(image_path, image_keys)
labels = get_labels(paths, gs.get_soh_labels())
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
  def __init__(self, img_dir=image_path, img_keys=image_keys, transform=None):
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
    image = Image.open(img_path)

    if self.transform:
      image = self.transform(image)

    return image, label
