import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ViT.model as M
import data.get_dataset as gds

config = {
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "lr": 1e-4,
  "epochs": 50,
  "batch_size": 32,
  "num_workers": 8,
  "weight_decay": 0.05,
  "save_path": "./best_model.pth"
}
# train_dataset = gds.BatteryDataset(
#   img_dir=r"F:\New\Coding\GAF_VIT\images",
#   img_keys=gds.train_image_keys,
#   transform=gds.get_transform()
# )
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataset = gds.BatteryDataset(
#   img_dir=r"F:\New\Coding\GAF_VIT\images",
#   img_keys=gds.val_image_keys,
#   transform=gds.get_transform()
# )
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
train_loader, val_loader, test_loader = gds.create_loaders()

def train():
  model = M.VisionTransformer(
    img_size=128,
    patch_size=16,
    in_chans=3,
    class_dim=1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer="nn.LayerNorm"
  ).to(config["device"])

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"]
  )

  best_mae= float("inf")
  for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch{epoch + 1}/{config['epochs']}")

    for images, labels in progress_bar:
      images = images.to(config["device"])
      labels = labels.to(config["device"]).float()

      output = model(images)
      loss = criterion(output, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item() * images.size(0)
      progress_bar.set_postfix({"loss" : loss.item()})

    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    # correct = 0
    # total = 0
    with torch.no_grad():
      for images, labels in val_loader:
        images = images.to(config["device"])
        labels = labels.to(config["device"])

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())
        # _, predicted = outputs.max(1)
        # total += labels.size(0)
        # correct += predicted.eq(labels).sum().item()

    # train_loss = train_loss / len(train_loader.dataset)
    # val_loss = val_loss / len(val_loader.dataset)
    # val_acc = 100. * correct / total
    val_loss = val_loss / len(val_loader.dataset)
    outputs = torch.cat(all_outputs).squeeze()
    labels = torch.cat(all_labels)
    mae = (outputs - labels).abs().mean().item()
    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | MAE: {mae:.4f}")

    if mae < best_mae:
      best_mae = mae
      torch.save(model.state_dict(), config["save_path"])
      print(f"Saved new best model with MAE{mae:.4f}")


if __name__ == "__main__":

  train()
