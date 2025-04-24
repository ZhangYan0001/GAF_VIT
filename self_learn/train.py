import torch
import torch.nn as nn
import numpy as np
import timm
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import data.get_dataset as dg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ViT.model as vit

"""
  self learning simsiam model
"""


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

    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    view1 = simsiam_augment(img)
    view2 = simsiam_augment(img)

    return view1, view2


class SimSiamEncoder(nn.Module):
  def __init__(
    self,
    base_model = None,
    projection_dim=1024,
    feature_dim=1024,
  ):
    super(SimSiamEncoder, self).__init__()

    self.encoder = base_model

    out_dim = self.encoder.num_features

    self.projection_head = nn.Sequential(
      nn.Linear(out_dim, projection_dim),
      nn.BatchNorm1d(projection_dim),
      nn.ReLU(inplace=True),
      nn.Linear(projection_dim, projection_dim),
    )

    self.prediction_head = nn.Sequential(
      nn.Linear(projection_dim, feature_dim),
      nn.BatchNorm1d(feature_dim),
      nn.ReLU(inplace=True),
      nn.Linear(feature_dim, projection_dim),
    )

  def forward(self, x):
    features = self.encoder(x)
    z = self.projection_head(features)
    p = self.prediction_head(z)
    return z, p


class SimSiam(nn.Module):
  def __init__(self, encoder):
    super(SimSiam, self).__init__()
    self.encoder = encoder

  def forward(self, x1, x2):
    z1, p1 = self.encoder(x1)
    z2, p2 = self.encoder(x2)

    return p1, p2, z1, z2

  def loss_fn(self, p1, p2, z1, z2):
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    loss_1 = -F.cosine_similarity(p1, z2.detach(), dim=-1)
    loss_2 = -F.cosine_similarity(p2, z1.detach(), dim=-1)

    loss = (loss_1 + loss_2).mean()
    # print("loss_1 min/max:", loss_1.min().item(), loss_1.max().item())
    # print("loss_2 min/max:", loss_2.min().item(), loss_2.max().item())
    # print("loss total:", loss.item())

    return loss


config = {
  "datasets_path": "/home/shunlizhang/zy/xj_datasets",
  "batch_size": 32,
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "epochs": 70,
  "lr": 0.00625,
  "weight_decay": 1e-4,
  "momentum": 0.9,
  "feature_dim": 768,
  "projection_dim": 1024,
  "base_model": None,
}


def train_simsiam(model, data_loader, optimizer, device):
  model.train()
  total_loss = 0.0

  with tqdm(data_loader, desc="Training", unit="batch") as tepoch:
    for data in tepoch:
      view1, view2 = data
      view1, view2 = view1.to(device), view2.to(device)

      optimizer.zero_grad()

      p1, p2, z1, z2 = model(view1, view2)
      loss = model.loss_fn(p1, p2, z1, z2)

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      total_loss += loss.item()
      tepoch.set_postfix(loss=loss.item())

  return total_loss / len(data_loader)


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
    combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4
  )
  return loader


def train():
  vit_backbone = vit.ViTBackbone(
    img_size=224, patch_size=16, in_chans=3,
    embed_dim=768, depth=4, num_heads=12, mlp_ratio=4,
    drop_rate=0.1, attn_drop_rate=0.1
  ).to(config["device"])

  encoder = SimSiamEncoder(
    base_model=vit_backbone,
    projection_dim=config["projection_dim"],
    feature_dim=config["feature_dim"],
  ).to(device=config["device"])

  model = SimSiam(encoder).to(config["device"])
  train_loader = data_loader(npy_files=config["datasets_path"])

  optimizer = optim.SGD(
    model.parameters(),
    lr=config["lr"],
    momentum=config["momentum"],
    weight_decay=config["weight_decay"],
  )

  scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["epochs"], eta_min=0.001
  )

  best_loss = float("inf")

  for epoch in range(config["epochs"]):
    avg_loss = train_simsiam(model, train_loader, optimizer, config["device"])
    scheduler.step()

    print(
      f"Epoch [{epoch + 1}/{config['epochs']}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    if avg_loss < best_loss:
      best_loss = avg_loss
      backbone = model.encoder.encoder
      torch.save(
        backbone.state_dict(),
        "./best_model5.pth"
      )


"""
  SOH model prediction 
"""


def encoder_fine_tuning(model_path: str, encoder):
  checkpoint = torch.load(model_path)
  encoder.load_state_dict(checkpoint["encoder"])
  return encoder


class SOHPredictionModel(nn.Module):
  def __init__(self, encoder, feature_dim:int):
    super(SOHPredictionModel, self).__init__()
    self.encoder = encoder
    self.fc = nn.Linear(feature_dim, 1)

  def forward(self, x):
    feat = self.encoder(x)
    if isinstance(feat, tuple):
      feat = feat[0]
    return self.fc(feat)

def unfreeze_encoder_layers(encoder, num_layers_to_unfreeze):
    for name, param in encoder.named_parameters():
        if "blocks" in name:
            layer_num = int(name.split("blocks.")[1].split(".")[0])
            if layer_num >= (12 - num_layers_to_unfreeze):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

def train_soh_model(model, train_loader, optimizer, loss_fn, device):
  model.train()
  total_loss = 0.0
  with tqdm(train_loader, desc="Training SOH", unit="batch") as tepoch:
    for data in tepoch:
      img, soh = data
      img, soh = img.to(device), soh.to(device)

      optimizer.zero_grad()

      predictions = model(img).squeeze()
      loss = loss_fn(predictions, soh)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      tepoch.set_postfix(loss=loss.item())

  return total_loss / len(train_loader)


def evaluate_soh_model(model, val_loader, loss_fn, device):
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for data in val_loader:
      img, soh = data
      img, soh = img.to(device), soh.to(device)

      predictions = model(img).squeeze()
      loss = loss_fn(predictions, soh)

      total_loss += loss.item()

  return total_loss / len(val_loader)

def get_vit_layers(encoder):
    return list(encoder.blocks)

def train_soh(train_loader, val_loader):
  config = {
    "datasets_path": "/home/shunlizhang/zy/xj_datasets",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 200,
    "lr": 0.00001,
    "unfreeze_interval":40,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "feature_dim": 768,
    "projection_dim": 1024,
    "base_model": None,
    "pretrained_model": "./best_model5.pth",
    "save_model":"./best_soh_model5.pth",
    'vit_kwargs': {
      'img_size': 224,
      'patch_size': 16,
      'in_chans': 3,
      'embed_dim': 768,
      'depth': 4,
      'num_heads': 12,
      'mlp_ratio': 4,
      'drop_rate': 0.1,
      'attn_drop_rate': 0.1,
    }
  }

  # encoder = SimSiamEncoder(
  #   feature_dim=config["feature_dim"],
  #   projection_dim=config["projection_dim"],
  #   base_model=None,
  # ).to(config["device"])
  vit_backbone = vit.ViTBackbone(**config["vit_kwargs"]).to(config["device"])
  state = torch.load(config["pretrained_model"], map_location=config["device"])
  vit_backbone.load_state_dict(state)
  vit_backbone.eval()
  # encoder = vit.ViTBackbone()
  # encoder = encoder_fine_tuning(config["pretrained_model"], encoder)

  # encoder.load_state_dict(torch.load(config["pretrained_model"], map_location=config["device"]))
  # encoder.eval()

  for param in vit_backbone.parameters():
    param.requires_grad = False

  model = SOHPredictionModel(vit_backbone,768).to(config["device"])

  optimizer = optim.AdamW(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"],
  )
  loss_fn = nn.MSELoss()

  vit_layers = vit_backbone.vit.blocks
  total_layers = len(vit_layers)
  best_loss = float("inf")

  for epoch in range(config["epochs"]):
    # num_layer_to_unfreeze = epoch // config["unfreeze_every"]
    # unfreeze_encoder_layers(encoder.encoder, num_layer_to_unfreeze)
    if epoch % config["unfreeze_interval"] == 0:
        unfreeze_idx = epoch // config["unfreeze_interval"]
        if unfreeze_idx < total_layers:
            print(f"Epoch {epoch}: 解冻第 {unfreeze_idx} 层 Transformer")
            for param in vit_layers[total_layers - 1 - unfreeze_idx].parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config["lr"],
                weight_decay=config["weight_decay"]
            )

    train_loss = train_soh_model(
      model, train_loader, optimizer, loss_fn, config["device"]
    )
    val_loss = evaluate_soh_model(model, val_loader, loss_fn, config["device"])
    print(
      f"Epoch [{epoch + 1}/{config['epochs']} | Train Loss : {train_loss:.4f} | Val Loss {val_loss:.4f}"
    )
    if val_loss < best_loss:
      best_loss = val_loss
      torch.save(model.state_dict(), config["save_model"])


def predict_soh(model, test_loader, device):
  model.eval()
  predictions = []
  true_labels = []
  with torch.no_grad():
    for data in test_loader:
      img, label = data
      img, label = img.to(device), label.to(device)
      pred = model(img).squeeze().cpu().numpy()
      predictions.append(pred)
      true_labels.append(label.cpu().numpy())

  predictions = np.concatenate(predictions)
  true_labels = np.concatenate(true_labels)

  return predictions, true_labels


def evaluate_soh(predictions, true_labels):
  mse = mean_squared_error(true_labels, predictions)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(true_labels, predictions)

  # 避免除以0
  true_labels = np.array(true_labels)
  predictions = np.array(predictions)
  mask = true_labels != 0
  mape = (
    np.mean(np.abs((true_labels[mask] - predictions[mask]) / true_labels[mask])) * 100
  )

  r2 = r2_score(true_labels, predictions)

  print(f"Mean Squared Error (MSE):       {mse:.4f}")
  print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
  print(f"Mean Absolute Error (MAE):      {mae:.4f}")
  print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
  print(f"R² Score:                        {r2:.4f}")

  return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def load_model(model_path, device):
  # encoder = SimSiamEncoder(
  #   base_model=None,
  #   projection_dim=config["projection_dim"],
  #   feature_dim=768,
  # ).to(device)
  config = {
    'vit_kwargs': {
      'img_size': 224,
      'patch_size': 16,
      'in_chans': 3,
      'embed_dim': 768,
      'depth': 4,
      'num_heads': 12,
      'mlp_ratio': 4,
      'drop_rate': 0.1,
      'attn_drop_rate': 0.1,
    }
  }
  encoder = vit.ViTBackbone(**config["vit_kwargs"]).to(device)
  model = SOHPredictionModel(encoder,768).to(device)

  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint)
  model.eval()

  return model


import matplotlib.pyplot as plt


def plot_soh_predictions(
  predictions, true_labels, title="SOH Prediction vs True Value"
):
  plt.figure(figsize=(6, 6))
  plt.scatter(true_labels, predictions, alpha=0.7, edgecolors="k", label="Predictions")
  plt.plot([0, 1], [0, 1], "r--", label="Ideal (y = x)")

  plt.xlabel("True SOH")
  plt.ylabel("Predicted SOH")
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.xlim(0.8, 1)
  plt.ylim(0.8, 1)
  plt.gca().set_aspect("equal", adjustable="box")
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  # train()
  xj_path = "/home/shunlizhang/zy/Batch-1"
  train_loader, val_loader, test_loader = dg.create_loaders(
    xj_path, dg.xj_image_keys, loader_flag="XJ",batch_size=32
  )
  train_soh(train_loader, val_loader)
  model = load_model("/home/shunlizhang/zy/gaf_-vit/self_learn/best_soh_model5.pth", config["device"])
  predictions, true_labels = predict_soh(model, test_loader, config["device"])
  evaluate_soh(predictions, true_labels)
  plot_soh_predictions(predictions, true_labels)
