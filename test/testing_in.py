import json
import os
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from PIL import Image
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def to_2tuple(x):
  return tuple([x] * 2)


trunc_normal_ = init.trunc_normal_
zeros_ = init.zeros_
ones_ = init.ones_


# 无操作的网络层
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, input):
    return input


# 图像分块、Embedding
class PatchEmbed(nn.Module):
  def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=768):
    super().__init__()
    # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
    img_size = to_2tuple(img_size)
    patch_size = to_2tuple(patch_size)
    # 图像块的个数
    num_patches = (img_size[1] // patch_size[1]) * \
                  (img_size[0] // patch_size[0])
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = num_patches
    # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
    # 输入维度为3，输出维度为块向量长度
    # 与原文中：分块、展平、全连接降维保持一致
    # 输出为[B, C, H, W]
    self.proj = nn.Conv2d(
      in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

  def forward(self, x):
    B, C, H, W = x.shape
    assert H == self.img_size[0] and W == self.img_size[1], \
      f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
    # x = self.proj(x).flatten(2).transpose((0, 2, 1))
    x = self.proj(x)
    x = x.flatten(2)
    x = x.transpose(1, 2)

    return x


# Multi-head Attention
class Attention(nn.Module):
  def __init__(self,
               dim,
               num_heads=8,
               qkv_bias=False,
               qk_scale=None,
               attn_drop=0.1,
               proj_drop=0.):
    super().__init__()
    self.num_heads = num_heads
    assert dim % num_heads == 0, "dim 必须能被num_heads整除"
    self.head_dim = dim // num_heads
    self.scale = qk_scale or self.head_dim ** -0.5
    # 计算 q,k,v 的转移矩阵
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    trunc_normal_(self.qkv.weight, std=0.02)
    if qkv_bias:
      nn.init.constant_(self.qkv.bias, 0)

    self.attn_drop = nn.Dropout(attn_drop)
    # 最终的线性层
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)
    # 投影层初始化
    trunc_normal_(self.proj.weight, std=0.02)
    nn.init.constant_(self.proj.bias, 0)

  def forward(self, x):
    B, N, C = x.shape
    # 线性变换
    # qkv = self.qkv(x).reshape((batch_size, N, 3, self.num_heads, C //
    #                            self.num_heads)).permute((2, 0, 3, 1, 4))
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    # 分割 query key value
    q, k, v = qkv.unbind(0)
    # Scaled Dot-Product Attention
    # Matmul + Scale
    attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
    # SoftMax
    attn = F.softmax(attn, dim=-1)
    attn = self.attn_drop(attn)
    # Matmul
    x = torch.einsum("bhij,bhjd->bhid", attn, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    # 线性变换
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class Mlp(nn.Module):
  def __init__(self,
               in_features,
               hidden_features=None,
               out_features=None,
               act_layer=nn.GELU,
               drop=0.1,
               use_ln=False
               ):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or int(in_features * 4)
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.drop1 = nn.Dropout(drop)
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop2 = nn.Dropout(drop)
    # 可选层归一化
    self.norm = nn.LayerNorm(hidden_features) if use_ln else nn.Identity()
    # 初始化
    trunc_normal_(self.fc1.weight, std=0.02)
    nn.init.constant_(self.fc1.bias, 0)
    trunc_normal_(self.fc2.weight, std=0.02)
    nn.init.constant_(self.fc2.bias, 0)

  def forward(self, x):
    # 输入层：线性变换
    x = self.fc1(x)
    # 应用激活函数
    x = self.act(x)
    # 可选层归一化
    x = self.norm(x)
    # Dropout
    x = self.drop1(x)
    # 输出层：线性变换
    x = self.fc2(x)
    # Dropout
    x = self.drop2(x)
    return x


def drop_path(x, drop_prob=0., training=False):
  if drop_prob == 0. or not training:
    return x
  keep_prob = torch.tensor(1 - drop_prob)
  shape = (x.shape[0],) + (1,) * (x.ndim - 1)
  random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
  random_tensor = torch.floor(random_tensor)
  output = x.div(keep_prob) * random_tensor
  return output


class DropPath(nn.Module):
  def __init__(self, drop_prob: float = 0.):
    super().__init__()
    self.drop_prob = drop_prob

  def forward(self, x: torch.Tensor):
    return drop_path(x, self.drop_prob, self.training)

  def extra_repr(self) -> str:
    return f"drop_prob = {self.drop_prob}"


class Block(nn.Module):
  def __init__(self,
               dim,
               num_heads=8,
               mlp_ratio=4.,
               qkv_bias=False,
               qk_scale=None,
               drop=0.,
               attn_drop=0.,
               drop_path=0.,
               act_layer=nn.GELU,
               norm_layer=partial(nn.LayerNorm, eps=1e-5),
               ):
    super().__init__()
    self.norm1 = norm_layer(dim)
    # Multi-head self-attention
    self.attn = Attention(
      dim,
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      qk_scale=qk_scale,
      attn_drop=attn_drop,
      proj_drop=drop
    )
    # DropPath
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(
      in_features=dim,
      hidden_features=mlp_hidden_dim,
      act_layer=act_layer,
      drop=drop,
      use_ln=True
    )

  def forward(self, x):
    # Multi-head Self-attent , add, LayerNorm
    x = x + self.drop_path(self.attn(self.norm1(x)))
    # Feed Forward, Add, LayerNorm
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


class VisionTransformer(nn.Module):
  def __init__(
    self,
    img_size=128,
    patch_size=16,
    in_chans=3,
    embed_dim=512,
    depth=6,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.,
    norm_layer=partial(nn.LayerNorm, eps=1e-5),
    **kwargs
  ):
    super().__init__()
    self.num_features = self.embed_dim = embed_dim
    # 图片分块和降维，块大小为patch_size，最终块向量维度为768
    self.patch_embed = PatchEmbed(
      img_size=img_size,
      patch_size=patch_size,
      in_chans=in_chans,
      embed_dim=embed_dim,
    )
    # 分块数量
    num_patches = self.patch_embed.num_patches
    # 可学习的位置编码
    self.pos_embed = nn.Parameter(
      torch.zeros(1, num_patches + 1, embed_dim)
    )
    # 回归专用token
    self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

    # transformer
    self.blocks = nn.ModuleList([
      Block(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop_rate,
        attn_drop=attn_drop_rate,
        drop_path=dpr[i],
        norm_layer=norm_layer,
      ) for i in range(depth)
    ])
    self.norm = norm_layer(embed_dim)
    self.reg_head = nn.Sequential(
      nn.Linear(embed_dim, embed_dim // 2),
      nn.GELU(),
      nn.Linear(embed_dim // 2, 1)
    )
    trunc_normal_(self.pos_embed, std=0.02)
    trunc_normal_(self.reg_token, std=0.02)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=0.02)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  def forward_features(self, x):
    B = x.shape[0]
    # 将图片分块，并调整每个快向量的维度
    x = self.patch_embed(x)
    # 将class token与前面的分块进行拼接
    reg_tokens = self.reg_token.expand(B, -1, -1)
    x = torch.cat((reg_tokens, x), dim=1)
    # 将编码向量中加入位置编码
    x = x + self.pos_embed
    x = self.pos_drop(x)
    # 堆叠 transformer结构
    for blk in self.blocks:
      x = blk(x)
    # LayerNorm
    x = self.norm(x)
    # 提取分类tokens的输出
    return x[:, 0]

  def forward(self, x):
    # 获取图像特征
    x = self.forward_features(x)
    # 图像分类
    x = self.reg_head(x)
    x = x.squeeze(-1)
    return x


class CNNRegressor(nn.Module):
  def __init__(self,
               input_size=128,  # 输入图像尺寸
               in_channels=3,  # 输入通道数（GAF为单通道）
               dropout_rate=0.3):  # 正则化强度
    super().__init__()

    # 特征提取器
    self.features = nn.Sequential(
      # Block 1: 128x128 -> 64x64
      nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Dropout(dropout_rate),

      # Block 2: 64x64 -> 32x32
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Dropout(dropout_rate),

      # Block 3: 32x32 -> 16x16
      nn.Conv2d(128, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Dropout(dropout_rate),

      # Block 4: 16x16 -> 8x8
      nn.Conv2d(256, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Dropout(dropout_rate)
    )

    # 回归头
    self.regressor = nn.Sequential(
      nn.Linear(512 * 8 * 8, 1024),
      nn.ReLU(),
      nn.Dropout(dropout_rate),

      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Dropout(dropout_rate),

      nn.Linear(512, 1)
    )

    # 初始化权重
    self._init_weights()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)  # 展平特征图
    return self.regressor(x).squeeze(-1)


image_path = r"/home/shunlizhang/zy/images3"
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
  "XQ-16"
]

val_image_keys = [
  "XQ-18",
  "XQ-17"
]


def get_soh_dict(file_path: str):
  with open(file_path, "r", encoding="utf-8") as f:
    SOH_Labels = json.load(f)
  return dict(SOH_Labels)


def get_images_path(image_dir: str, image_keys: [str]):
  if not os.path.exists(image_dir):
    print("the image_dir is not exist,please input a new path")
  images = []
  for image_key in image_keys:
    path = pathlib.Path(os.path.join(image_path, image_key + "-images"))
    for P in path.iterdir():
      images.append(P.__str__())

  return images


def get_labels(image_paths: [], SOH_Labels: dict):
  labels = []
  for path in image_paths:
    key = "XQ-" + path.split("\\")[-1].split("-")[1]
    idx = path.split("\\")[-1].split("-")[-1].split(".")[0]
    label = SOH_Labels[key][idx]
    labels.append(label)
  return labels


# get_soh = get_soh_dict(r"F:\New\Coding\GAF_VIT\data\soh_data.json")
# print(get_soh)
# print(type(get_soh))
# labels = get_labels(get_images_path(image_keys), get_soh)
# print(labels)
class BatteryDataset(Dataset):
  def __init__(self, img_dir=image_path, img_keys=image_keys, labels=[], transform=None):
    self.img_dir = img_dir
    self.img_keys = img_keys
    self.transform = transform
    self.path_data = get_images_path(img_dir, img_keys)
    self.labels = labels

  def __len__(self):
    return len(self.path_data)

  def __getitem__(self, index):
    img_path = self.path_data[index]
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
  SOH_Labels = get_soh_dict(r"/home/shunlizhang/zy/gaf_-vit/data/soh_data.json")
  all_paths = get_images_path(image_path, image_keys)
  all_labels = get_labels(all_paths, SOH_Labels)

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


config = {
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "lr": 1e-4,
  "epochs": 50,
  "batch_size": 32,
  "num_workers": 8,
  "weight_decay": 0.05,
  "save_path": "./best_vit_model.pth"
}

train_loader, val_loader, test_loader = create_loaders()


def train_vit():
  model = VisionTransformer(
    img_size=128,
    patch_size=16,
    in_chans=3,
    embed_dim=96,
    depth=4,
    num_heads=6,
    mlp_ratio=2,
    qkv_bias=True,
    drop_rate=0.2,
    attn_drop_rate=0.2,
    norm_layer=partial(nn.LayerNorm, eps=1e-5)
  ).to(config["device"])

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"]
  )
  warmup_epochs = 5
  scheduler_warmup = LinearLR(
    optimizer,
    start_factor=0.01,
    total_iters=warmup_epochs
  )
  scheduler_cos = CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"] - warmup_epochs
  )
  scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_cos],
    milestones=[warmup_epochs]
  )

  best_r2 = -float("inf")
  early_stop_counter = 0

  for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch{epoch + 1}/{config['epochs']}")

    for images, labels in progress_bar:
      images = images.to(config["device"], dtype=torch.float32)
      labels = labels.to(config["device"], dtype=torch.float32)

      with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(images)
        loss = criterion(output.squeeze(), labels)

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item() * images.size(0)
      progress_bar.set_postfix({"loss": loss.item()})

    scheduler.step()

    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
      for images, labels in val_loader:
        images = images.to(config["device"], dtype=torch.float32)
        labels = labels.to(config["device"], dtype=torch.float32)

        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        val_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

    val_loss = val_loss / len(val_loader.dataset)
    outputs = torch.cat(all_outputs).squeeze()
    labels = torch.cat(all_labels)
    mae = (outputs - labels).abs().mean().item()
    r2 = r2_score(labels.numpy(), outputs.numpy())
    print(
      f"Epoch {epoch + 1} | "
      f"Val Loss: {val_loss:.4f} | "
      f"MAE : {mae:.4f} | "
      f"LR: {optimizer.param_groups[0]['lr']:.2f} | "
    )
    if r2 > best_r2:
      best_r2 = r2
      torch.save(model.state_dict(), config["save_path"])
      early_stop_counter = 0
      print("Saved the model ")
    else:
      early_stop_counter += 1
      if early_stop_counter >=10:
        print("Early stopping triggered!")
        break

  print(f"Best R2 : {best_r2:.4f}")

def train_cnn():
  config_cnn = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
    "weight_decay": 0.05,
    "epochs": 50
  }
  model = CNNRegressor(
    input_size=128,
    in_channels=3,
    dropout_rate=0.3
  ).to(config_cnn["device"])
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config_cnn["lr"],
    weight_decay=config_cnn["weight_decay"]
  )
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    verbose=True
  )

  best_mae = float("inf")
  for epoch in range(config_cnn["epochs"]):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch{epoch + 1}/{config_cnn['epochs']}")

    for images, labels in progress_bar:
      images = images.to(config_cnn["device"])
      labels = labels.to(config_cnn["device"]).float()

      optimizer.zero_grad()
      output = model(images)
      loss = criterion(output, labels)

      if torch.isnan(loss):
        print("Warning: NaN loss detected, skipping batch")
        continue

      loss.backward()
      optimizer.step()

      train_loss += loss.item() * images.size(0)
      progress_bar.set_postfix({"loss": loss.item()})

    train_loss /= len(train_loader.dataset)
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
      for images, labels in val_loader:
        images = images.to(config_cnn["device"])
        labels = labels.to(config_cnn["device"])

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

    val_loss = val_loss / len(val_loader.dataset)
    outputs = torch.cat(all_outputs).squeeze()
    labels = torch.cat(all_labels)
    mae = (outputs - labels).abs().mean().item()
    print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | MAE: {mae:.4f}")

    scheduler.step(val_loss)

    if mae < best_mae:
      best_mae = mae
      torch.save(model.state_dict(), config_cnn["save_path"])
      print(f"Saved the model with MAE: {best_mae:.4f}")

  print("Training complete.")


def evaluate_model(model, test_loader, checkpoint_path="best_model.pth", device=None):
  """
  使用训练好的 CNN 模型进行测试并评估性能。

  参数：
  - model: 训练好的 CNN 模型
  - test_loader: 测试数据 DataLoader
  - checkpoint_path: 训练好的模型权重文件路径
  - device: 运行设备（自动检测 GPU 或 CPU）
  """
  # 自动选择设备
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  model.to(device)
  model.eval()

  # 加载训练好的模型权重
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  print(f"Loaded model from {checkpoint_path}")

  criterion = nn.MSELoss()  # 计算 MSE 误差
  all_outputs, all_labels = [], []
  total_loss = 0.0

  with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
      images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)
      total_loss += loss.item() * images.size(0)

      all_outputs.append(outputs.cpu().numpy().astype(np.float32))
      all_labels.append(labels.cpu().numpy().astype(np.float32))

  # 计算整体误差
  test_loss = total_loss / len(test_loader.dataset)
  outputs = np.concatenate(all_outputs, axis=0)
  labels = np.concatenate(all_labels, axis=0)

  # 计算 MAE
  mae = np.mean(np.abs(outputs - labels))

  # 计算 RMSE (均方根误差)
  rmse = np.sqrt(np.mean((outputs - labels) ** 2))

  # 计算 R² Score (决定系数)
  r2 = r2_score(labels, outputs)

  print(f"Test Loss (MSE): {test_loss:.4f}")
  print(f"MAE: {mae:.4f}")
  print(f"RMSE: {rmse:.4f}")
  print(f"R² Score: {r2:.4f}")

  return outputs, labels, rmse, r2


def plot_results(outputs, labels, num_samples=100):
  """
  绘制预测值与真实值对比图。

  参数：
  - outputs: 预测值 (Tensor)
  - labels: 真实值 (Tensor)
  - num_samples: 要显示的样本数量（默认 100 个）
  """
  # outputs = outputs.numpy()
  # labels = labels.numpy()

  plt.figure(figsize=(10, 5))
  plt.plot(labels[:num_samples], label="True Values", marker='o', linestyle="--")
  plt.plot(outputs[:num_samples], label="Predicted Values", marker='s', linestyle="-")

  plt.xlabel("Sample Index")
  plt.ylabel("Value")
  plt.legend()
  plt.title("CNN Regression Results")
  plt.grid(True)
  plt.show()


if __name__ == "__main__":
  train_vit()
  #
  # # train_cnn()
  print("this the cnn model evaluate result: ")
  cnn_outputs, cnn_labels, cnn_rmse, cnn_r2 = evaluate_model(
    CNNRegressor(),
    test_loader,
    "./best_cnn_model.pth"
  )
  plot_results(cnn_outputs, cnn_labels)

  print("this the vit model evaluate result: ")
  vit_outputs, vit_labels, vit_rmse, vit_r2 = evaluate_model(
    VisionTransformer(img_size=128,
    patch_size=16,
    in_chans=3,
    embed_dim=96,
    depth=4,
    num_heads=6,
    mlp_ratio=2,
    qkv_bias=True,),
    test_loader,
    "./best_vit_model.pth"
  )
  plot_results(vit_outputs, vit_labels)
