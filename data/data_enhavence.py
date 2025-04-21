import numpy as np
import torch
from typing import Union
import os
from pathlib import Path
from scipy.interpolate import interp1d
import data.get_feature as gf
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import ConcatDataset
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from data.tjbattery import TJBattery
from data.xjbattery import Battery
from pyts.image import GramianAngularField


class TimeSeriesAugmentor:
  def __init__(self, target_length=224, noise_std=0.01):
    self.target_length = target_length
    self.noise_std = noise_std

  def add_noise(self, ts):
    ts = np.array(ts)
    std = np.std(ts)
    noise = np.random.normal(0, self.noise_std * std, size=ts.shape)
    return noise + ts

  def time_scale(self, ts, scale_factor=None):
    if scale_factor is None:
      scale_factor = np.random.uniform(0.8, 1.2)

    old_len = len(ts)
    new_len = int(old_len * scale_factor)

    x_old = np.linspace(0, 1, old_len)
    x_new = np.linspace(0, 1, new_len)
    f = interp1d(x_old, ts, kind="linear")
    ts_scaled = f(x_new)

    return ts_scaled

  def resize(self, ts):
    curr_len = len(ts)
    x_old = np.linspace(0, 1, curr_len)
    x_new = np.linspace(0, 1, self.target_length)
    f = interp1d(x_old, ts, kind="linear")
    return f(x_new)

  def augment_pair(self, ts):
    ts1 = self.add_noise(ts)
    ts1 = self.resize(ts1)

    ts2 = self.time_scale(ts)
    ts2 = self.resize(ts2)

    return ts1, ts2


class SimSiamGAF3Augmentor:
  def __init__(
    self, image_size=224, drop_prob=0.3, patch_noise_prob=0.3, patch_size=16
  ):
    self.base_Transform = T.Compose(
      [
        T.RandomResizedCrop(
          image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.GaussianBlur(kernel_size=3),
      ]
    )
    self.drop_prob = drop_prob
    self.patch_noise_prob = patch_noise_prob
    self.patch_size = patch_size

  def drop_channel(self, img):
    if np.random.rand() < self.drop_prob:
      c = np.random.choice(3)
      img[c] = 0.0

    return img

  def add_patch_noise(self, img):
    if np.random.rand() < self.patch_noise_prob:
      C, H, W = img.shape
      x = np.random.randint(0, H - self.patch_size)
      y = np.random.randint(0, W - self.patch_size)
      noise = (
        torch.randn_like(img[:, x : x + self.patch_size, y : y + self.patch_size]) * 0.1
      )
      img[:, x : x + self.patch_size, y : y + self.patch_size] += noise

    return img

  def __call__(self, img):
    img = T.ToPILImage()(img)
    img = self.base_Transform(img)
    img = T.ToTensor()(img)
    img = self.drop_channel(img)
    img = self.add_patch_noise(img)

    return img


def simsiam_augment(img, image_size=224):
  base_transform = T.Compose(
    [
      T.RandomResizedCrop(
        image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC
      ),
      T.RandomHorizontalFlip(p=0.5),
      T.RandomRotation(10),
      T.ColorJitter(brightness=0.2, contrast=0.2),
      T.GaussianBlur(kernel_size=3),
    ]
  )

  def drop_channel(img, drop_prob=0.3):
    if np.random.rand() < drop_prob:
      c = np.random.choice(3)
      img[c] = 0.0
    return img

  def add_patch_noise(img, patch_noise_prob=0.3, patch_size=16):
    if np.random.rand() < patch_noise_prob:
      C, H, W = img.shape
      x = np.random.randint(0, H - patch_size)
      y = np.random.randint(0, W - patch_size)
      noise = torch.randn_like(img[:, x : x + patch_size, y : y + patch_size]) * 0.1
      img[:, x : x + patch_size, y : y + patch_size] += noise
    return img

  img = T.ToPILImage()(img)
  img = base_transform(img)
  img = T.ToTensor()(img)
  img = drop_channel(img)
  img = add_patch_noise(img)
  return img


class GAF3SimSiamDataset(torch.utils.data.Dataset):
  def __init__(self, image_list):
    self.image_list = image_list
    # self.augmentor = SimSiamGAF3Augmentor()

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    img = self.image_list[idx]
    if isinstance(img, str):
      img = Image.open(img)
      img = img.convert("RGB")
      img = img.resize((224, 224))
      img = np.array(img)
      # img = np.load(img)

    img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
    view1 = simsiam_augment(img)
    view2 = simsiam_augment(img)

    return view1, view2


def show_augmented_pair(original, view1, view2):
  fig, axs = plt.subplots(1, 3, figsize=(12, 4))
  titles = ["Original", "View 1", "View 2"]
  imgs = [original, view1, view2]

  for i in range(3):
    img_np = imgs[i].detach().cpu().numpy().transpose(1, 2, 0)
    axs[i].imshow(img_np)
    axs[i].set_title(titles[i])
    axs[i].axis("off")

  plt.tight_layout()
  plt.show()


def save_to_npy(data, filename):
  np.save(filename, data)


def gaf(data):
  scaler = MinMaxScaler(feature_range=(0, 1))
  data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
  # print("data normalized: ", data_normalized)
  image_size = data.size

  gaf_data = GramianAngularField(
    image_size=image_size, method="summation", sample_range=(0, 1)
  )
  gaf_data = np.array(gaf_data.fit_transform(data_normalized.reshape(1, -1)))[0]
  gaf_data = (gaf_data + 1.0) / 2.0

  return gaf_data


def fusion(data1, data2, data3):
  fusion_data = np.stack([data1, data2, data3], axis=-1)
  return fusion_data


def gaf3_data(battery: Union[Battery, TJBattery], stage=1, batch="", save_path=None):
  fusion_datas = []
  print(battery.battery_name)

  if isinstance(battery, Battery):
    bdata_ = gf.get_xj_battery_data(battery, stage)
  elif isinstance(battery, TJBattery):
    bdata_ = gf.get_tj_battery_data(battery)
  else:
    print("the battery is unknown")
    return None

  for cycle_data in bdata_:
    vol = cycle_data["voltage"]
    cur = cycle_data["current"]
    rel_vol = cycle_data["charge_rel_voltage"]
    # print(f"current cycle: {cycle_data['cycle']}")
    if len(rel_vol) < 2:
      continue

    if isinstance(battery, TJBattery):
      vol, cur, rel_vol = (
        gf.resample(vol, 224),
        gf.resample(cur, 224),
        gf.resample(rel_vol, 224),
      )

    vol_, cur_, rel_vol_ = gaf(vol), gaf(cur), gaf(rel_vol)
    fusion_data = fusion(vol_, cur_, rel_vol_)
    # augmentor = SimSiamGAF3Augmentor()
    # augmented_fusion_data = augmentor(fusion_data)
    # augmented_fusion_data = augmented_fusion_data.permute(1, 2, 0)
    fusion_datas.append(fusion_data)

  if save_path is not None:
    save_to_npy(
      np.array(fusion_datas), save_path + "/" + f"{batch}_{battery.battery_name}.npy"
    )

  return fusion_datas


# def load_from_npy(filename):
#   return np.load(filename)


# def save_batch_data_to_npy(batch_data, batch_idx, data_type="tj"):
#   filename = f"D:/1New/Coding/GAF_VIT/datasets/{data_type}_batch_{batch_idx}.npy"
#   save_to_npy(batch_data, filename)


def gaf3_all_data(save_path):
  for batch in gf.tj_batches_arr:
    tj_data = gf.get_tj_all_datas(batch)
    for _, battery in tj_data.items():
      gaf3_data(battery, batch=batch, save_path=save_path)

  for batch in gf.batches_arr:
    xj_data = gf.get_xj_batch_data(batch)
    for _, battery in xj_data.items():
      gaf3_data(battery, batch=batch, save_path=save_path)


def load_all_npy_data(npy_files):
  all_data = []
  for npy_file in npy_files:
    data = np.load(npy_file)
    all_data.append(data)

  return all_data


# todo 重构代码 重新实现 加载数据集 保存npy
def data_loader(batch_size=32, image_size=224, npy_files=None):
  # gaf3_all_data(save_path=npy_files)
  files_list = [npy_files + "/" + path for path in os.listdir(npy_files)]
  datasets = []
  for file in files_list:
    data = np.load(file)
    dataset = GAF3SimSiamDataset(data)
    datasets.append(dataset)

  combined_dataset = ConcatDataset(datasets)
  loader = DataLoader(
    combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0
  )
  return loader


"""
  testing
"""

# if __name__ == "__main__":
# gaf3_all_data()
# gaf3_all_data()
# loader = data_loader(npy_files=r"D:/1New/Coding/GAF_VIT/xj_datasets")
# battery_datas1 = gf.get_tj_all_datas("Dataset_1_NCA_battery")
# for k, battery in battery_datas1.items():
#   print(k)
#   print(battery)
#   battery_data = gf.get_tj_battery_data(battery)
#   data_augmentor = TimeSeriesAugmentor()
#   for cycle_data in battery_data:
#     ch_rel_voltage = cycle_data["charge_rel_voltage"]
#     ch1, ch2 = data_augmentor.augment_pair(ch_rel_voltage)
#     print(f"the ch1:{ch1}, and the len: {len(ch1)}")
#     print(f"the ch2:{ch2}, and the len: {len(ch2)}")
#     break
#   break
# data_loader()
