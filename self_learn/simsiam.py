import torch
import torch.nn as nn
import timm
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import data.data_enhavence as de


class SimSiamEncoder(nn.Module):
  def __init__(
    self,
    base_model="vit_base_patch16_224",
    projection_dim=2048,
    feature_dim=2048,
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

    return loss


config = {
  "datasets_path":"/home/zy/xj_datasets",
  "batch_size": 32,
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "epochs": 10,
  "lr": 0.05,
  "weight_decay": 1e-4,
  "momentum": 0.9,
  "feature_dim": 2048,
  "projection_dim": 2048,
  "base_model": "vit_base_patch16_224",
}


def train_simsiam(model, data_loader, optimizer, device):
  model.train()
  total_loss = 0.0

  with tqdm(data_loader, desc="Training", unit="batch") as tepoch:
    for data in data_loader:
      view1, view2 = data
      view1, view2 = view1.to(device), view2.to(device)

      optimizer.zero_grad()

      p1, p2, z1, z2 = model(view1, view2)
      loss = model.loss_fn(p1, p2, z1, z2)

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      # loss.backward()
      # optimizer.setp()

      total_loss += loss.item()
      tepoch.set_postfix(loss=loss.item())

  return total_loss / len(data_loader)


def train():
  encoder = SimSiamEncoder(
    base_model=config["base_model"],
    projection_dim=config["projection_dim"],
    feature_dim=config["feature_dim"],
  ).to(device=config["device"])

  model = SimSiam(encoder).to(config["device"])
  train_loader = de.data_loader(npy_files=config["datasets_path"])

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
        "./best_model.pth",
      )


if __name__ == "__main__":
  train()