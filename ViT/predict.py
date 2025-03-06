import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchvision import transforms

import ViT.model as vit
import data.get_dataset as gds


def evaluate_model(model, test_loader, device):
  model.eval()
  true_labels = []
  pred_labels = []

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.cpu().numpy()

      outputs = model(images).cpu().numpy().flatten()

      true_labels.extend(labels)
      pred_labels.extend(outputs)

  true_labels = np.array(true_labels)
  pred_labels = np.array(pred_labels)

  mae = mean_absolute_error(true_labels, pred_labels)
  rmse = np.sqrt(mean_squared_error(true_labels, pred_labels))
  r2 = r2_score(true_labels, pred_labels)

  print(f"MAE: {mae:.4f}")
  print(f"RMSE: {rmse:.4f}")
  print(f"R2 Score: {r2:.4f}")

  plt.figure(figsize=(10, 6))
  plt.scatter(true_labels, pred_labels, alpha=0.5)
  plt.plot([min(true_labels), max(true_labels)],
           [min(true_labels), max(true_labels)], "r--")
  plt.xlabel("True Values")
  plt.ylabel("Predictions")
  plt.title("True vs Predicted Values")
  plt.show()

  return mae, rmse


def predict_single_image(model, image_path, transform, device):
  image = Image.open(image_path).convert("L")
  image_tensor = transform(image).unsqueeze(0).to(device)
  model.eval()
  with torch.no_grad():
    prediction = model(image_tensor).cpu().item()

  return prediction


if __name__ == "__main__":
  config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": r"F:\New\Coding\GAF_VIT\result\best_vit_model3.pth",
    "image_size": 128
  }
  test_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=0.45, std=0.2)
  ])
  model =vit.VisionTransformer(
    img_size=config["image_size"],
    patch_size=16,
    in_chans=1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
  ).to(config["device"])
  model.load_state_dict(torch.load(config["model_path"]))

  _, _, test_loader = gds.create_loaders()
  print("Evaluating on test set:")

  evaluate_model(model, test_loader, config["device"])
  sample_image_path = r"F:\New\Coding\GAF_VIT\images3\XQ-15-images\XQ-15-789.png"
  prediction = predict_single_image(
    model=model,
    image_path=sample_image_path,
    transform=test_transform,
    device=config["device"]
  )
  print(f"Predicted SOH: {prediction:.4f}")
