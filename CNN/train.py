import torch
import torch.nn as nn
import torch.optim as optim
import CNN.cnn as cnn
from tqdm import tqdm
import data.get_dataset as gds

train_loader, val_loader, test_loader = gds.create_loaders()

def train_cnn():
  config_cnn = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
    "weight_decay": 0.05,
    "epochs": 50,
    "save_path":"./best_cnn_model3.pth"
  }
  model = cnn.CNNRegressor(
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