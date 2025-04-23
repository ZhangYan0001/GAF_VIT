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
from sklearn.metrics import mean_squared_error, r2_score


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
    base_model="vit_base_patch16_224",
    projection_dim=1024,
    feature_dim=1024,
  ):
    super(SimSiamEncoder, self).__init__()
    self.encoder = timm.create_model(base_model, pretrained=True, num_classes=0)
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
  "epochs": 30,
  "lr": 0.0001,
  "weight_decay": 1e-4,
  "momentum": 0.9,
  "feature_dim": 1024,
  "projection_dim": 1024,
  "base_model": "vit_base_patch16_224",
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
  encoder = SimSiamEncoder(
    base_model=config["base_model"],
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
      torch.save(
        {
          "encoder": encoder.state_dict(),
          "optimizer": optimizer.state_dict(),
          "epoch": epoch,
          "loss": avg_loss,
        },
        "./best_model4.pth",
      )


"""
  SOH model prediction 
"""


def encoder_fine_tuning(model_path: str, encoder):
  checkpoint = torch.load(model_path)
  encoder.load_state_dict(checkpoint["encoder"])
  return encoder


class SOHPredictionModel(nn.Module):
  def __init__(self, encoder):
    super(SOHPredictionModel, self).__init__()
    self.encoder = encoder
    self.fc = nn.Linear(config["feature_dim"], 1)

  def forward(self, x):
    z, _ = self.encoder(x)
    soh_pred = self.fc(z)
    return soh_pred


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


def train_soh(train_loader, val_loader):
  config = {
    "datasets_path": "/home/shunlizhang/zy/xj_datasets",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 100,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "feature_dim": 1024,
    "projection_dim": 1024,
    "base_model": "vit_base_patch16_224",
    "pretrained_model": "./best_model2.pth",
  }

  encoder = SimSiamEncoder(feature_dim=1024,projection_dim=config["projection_dim"],base_model=config["base_model"]).to(config["device"])
  encoder = encoder_fine_tuning(config["pretrained_model"], encoder)

  for param in encoder.parameters():
    param.requires_grad = False

  model = SOHPredictionModel(encoder).to(config["device"])

  optimizer = optim.Adam(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"],
  )
  loss_fn = nn.MSELoss()

  best_loss = float("inf")
  for epoch in range(config["epochs"]):
    train_loss = train_soh_model(
      model, train_loader, optimizer, loss_fn, config["device"]
    )
    val_loss = evaluate_soh_model(model, val_loader, loss_fn, config["device"])
    print(
      f"Epoch [{epoch + 1}/{config['device']} | Train Loss : {train_loss:.4f} | Val Loss {val_loss:.4f}"
    )
    if val_loss < best_loss:
      best_loss = val_loss
      torch.save(model.state_dict(), "./best_soh_model.pth")


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
  print(f"Mean Squared Error(MSE): {mse:.4f}")

  r2 = r2_score(true_labels, predictions)
  print(f"R2 Score: {r2:.4f}")

  return mse, r2


def load_model(model_path, device):
  encoder = SimSiamEncoder(
    base_model=config["base_model"],
    projection_dim=config["projection_dim"],
    feature_dim=config["feature_dim"],
  ).to(device)
  model = SimSiam(encoder).to(device)

  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint["model_state_dict"])

  return model

def evaluate_soh(predictions, true_labels):
  mse = mean_squared_error(true_labels, predictions)
  print(f"Mean Squared Error (MSE): {mse:.4f}")
  r2 = r2_score(true_labels, predictions)
  print(f"R² Score: {r2:.4f}")

  return mse, r2

if __name__ == "__main__":
  # train()
  xj_path = "/home/shunlizhang/zy/Batch-1"
  train_loader, val_loader, test_loader = dg.create_loaders(
    xj_path, dg.xj_image_keys, loader_flag="XJ"
  )
  # train_soh(train_loader, val_loader)
  model = load_model("", config["device"])
  predictions, true_labels = predict_soh(model, test_loader, config["device"])
  mse, r2 = evaluate_soh(predictions, true_labels)
