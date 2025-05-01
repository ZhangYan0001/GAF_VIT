import re

import torch
from datetime import datetime
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import pathlib

import data.get_dataset as dg
import data.get_feature as gf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import matplotlib.pyplot as plt
import ViT.model as vit


class SOHPredictionModel(nn.Module):
  def __init__(self, encoder, feature_dim: int):
    super(SOHPredictionModel, self).__init__()
    self.encoder = encoder
    self.fc = nn.Linear(feature_dim, 1)

  def forward(self, x):
    feat = self.encoder(x)
    if isinstance(feat, tuple):
      feat = feat[0]
    return self.fc(feat)


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


def load_model(model_path, device, config):
  encoder = vit.ViTBackbone(**config["vit_kwargs"]).to(device)
  model = SOHPredictionModel(encoder, config["vit_kwargs"]["embed_dim"]).to(device)

  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint)
  model.eval()

  return model


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

  return {
    "MSE": float(mse),
    "RMSE": float(rmse),
    "MAE": float(mae),
    "MAPE": float(mape),
    "R2": float(r2),
  }


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


class OneBatteryDataset(Dataset):
  def __init__(self, img_dir, labels, transform):
    self.img_dir = img_dir
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.img_dir)

  def __getitem__(self, item):
    img_path = self.img_dir[item]
    label = self.labels[item]
    image = Image.open(img_path).convert("RGB")
    label = torch.tensor(label, dtype=torch.float)

    if self.transform:
      image = self.transform(image)

    return image, label


def get_batch_xj_data(batch):
  return gf.get_xj_batch_data(batch)


def battery_one_data_loader(path, key, datas):
  image_list = [os.path.join(path, image_path) for image_path in os.listdir(path)]

  def extract_number(fname):
    match = re.search(r"-(\d+)\.png", fname)
    return int(match.group(1)) if match else -1

  image_list_sorted = sorted(image_list, key=extract_number)

  label_list = gf.cal_xj_battery_soh_data(datas[key])
  battery_dataset = OneBatteryDataset(
    img_dir=image_list_sorted,
    labels=list(label_list.values())[1:],
    transform=dg.get_val_transform(),
  )

  battery_loader = DataLoader(
    battery_dataset, batch_size=32, shuffle=False, num_workers=4
  )
  return battery_loader


if __name__ == "__main__":
  #
  config_simsiam = {
    "datasets_path": "/home/shunlizhang/zy/xj_datasets",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 200,
    "lr": 0.007,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "feature_dim": 1024,
    "projection_dim": 1024,
    "base_model": None,
    "save_model": "./best_model9.pth",
    "vit_kwargs": {
      "img_size": 224,
      "patch_size": 16,
      "in_chans": 3,
      "embed_dim": 1024,
      "depth": 6,
      "num_heads": 16,
      "mlp_ratio": 4,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
    },
  }
  config = {
    "datasets_path": "/home/shunlizhang/zy/xj_datasets",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 250,
    "lr": 0.0002,
    "unfreeze_interval": 50,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "feature_dim": 1024,
    "projection_dim": 1024,
    "base_model": None,
    "pretrained_model": "./best_model9.pth",
    "save_model": "./best_soh_model9.pth",
    "vit_kwargs": {
      "img_size": 224,
      "patch_size": 16,
      "in_chans": 3,
      "embed_dim": 1024,
      "depth": 6,
      "num_heads": 16,
      "mlp_ratio": 4,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
    },
  }
  batch_1_datas = get_batch_xj_data("Batch-1")
  now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  all_config = {
    now: {
      "config": config,
      "config_simsiam": config_simsiam,
    }
    # "param_space_simsiam": param_space,
  }

  # battery_loader = battery_one_data_loader()

  # train(config_simsiam)
  xj_path = "D:/1New/Coding/GAF_VIT/fusion_images/Batch-1"
  tj_path = "D:/1New/Coding/GAF_VIT/fusion_images/Dataset_1_NCA_battery"

  img_lists = [os.path.join(xj_path, key) for key in os.listdir(xj_path)]
  print(img_lists)

  battery_loaders = {}

  for img_list in img_lists:
    key = img_list.split("\\")[-1]
    battery_loader = battery_one_data_loader(img_list, key, batch_1_datas)
    battery_loaders[key] = battery_loader

  #   train_loader, val_loader, test_loader = dg.create_loaders(
  #   xj_path, dg.xj_image_keys, loader_flag="XJ", batch_size=32
  # )

  # tj_train, tj_val, tj_test = dg.create_loaders(
  #   tj_path, dg.tj_image_keys, loader_flag="TJ", batch_size=32
  # )
  # train_soh(train_loader, val_loader, config=config)
  # train_soh(tj_train, tj_val, config=config)
  model = load_model(
    "./best_soh_model20.pth",
    config["device"],
    config,
  )


  for key, loader in battery_loaders.items():
    # loader  = battery_one_data_loader()
    results_json = {}
    pred, true = predict_soh(model, loader, config["device"])
    result = evaluate_soh(pred, true)
    plot_soh_predictions(pred, true)
    results_json[key] = {}
    results_json[key]["predict"] = pred.tolist()
    results_json[key]["true"] = true.tolist()
    results_json[key]["result"] = result

    with open("all_xj_batch_1_results.jsonl", "a", encoding="utf-8") as f:
      f.write(json.dumps(results_json, ensure_ascii=False) + "\n")
  #
  # predictions, true_labels = predict_soh(model, test_loader, config["device"])
  # # tj_predict, tj_true_l = predict_soh(model, tj_test, config["device"])
  #
  # result = evaluate_soh(predictions, true_labels)
  # # result = evaluate_soh(tj_predict, tj_true_l)
  #
  # plot_soh_predictions(predictions, true_labels)
  # # plot_soh_predictions(tj_predict,tj_true_l)
  # all_config["result"] = result
  #
  # with open("./best_model_soh&simsiam.jsonl", "a") as f:
  #   f.write(json.dumps(all_config, ensure_ascii=False))
  #   f.write("\n")
