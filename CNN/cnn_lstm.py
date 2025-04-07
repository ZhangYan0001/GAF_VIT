import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CnnLstmRegressor(nn.Module):
  def __init__(
    self, img_size=32, time_steps=10, num_channels=1, lstm_units=64, fc_units=64
  ):
    super(CnnLstmRegressor, self).__init__()
    self.time_steps = time_steps
    self.cnn = nn.Sequential(
      nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )
    cnn_output_dim = 64 * (img_size // 4) ** 2
    self.lstm = nn.LSTM(
      input_size=cnn_output_dim,
      hidden_size=lstm_units,
      batch_first=True,
    )

    self.fc1 = nn.Linear(lstm_units, fc_units)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(fc_units, 1)

  def forward(self, x):
    batch_size, time_steps, C, H, W = x.size()
    x = x.view(batch_size * time_steps, C, H, W)
    x = self.cnn(x)
    x = x.view(batch_size, time_steps, -1)
    x, _ = self.lstm(x)
    x = self.dropout(F.relu(self.fc1(x[:, -1, :])))
    return self.fc2(x)


# 1. 模拟数据集类
class SyntheticTimeSeriesImages(Dataset):
  def __init__(self, num_samples=1000, img_size=32, time_steps=10):
    self.num_samples = num_samples
    self.img_size = img_size
    self.time_steps = time_steps
    self.data = []
    self.labels = []

    # 生成模拟数据
    for _ in range(num_samples):
      # 生成基础趋势 (线性+正弦)
      trend = np.linspace(0, np.random.uniform(-2, 2), time_steps)
      seasonality = np.sin(np.linspace(0, np.random.randint(3, 8) * np.pi, time_steps))

      # 生成时间序列值并归一化到[0,1]
      values = trend + seasonality
      values = (values - values.min()) / (values.max() - values.min())

      # 为每个时间步创建"图像" (模拟GAF图像)
      sequence = []
      for t in range(time_steps):
        # 创建对角占优的矩阵模拟GAF
        img = np.zeros((img_size, img_size))
        for i in range(img_size):
          for j in range(img_size):
            # 对角元素反映当前时间步值
            if i == j:
              img[i, j] = values[t]
            # 非对角元素反映时间相关性
            else:
              dist = abs(i - j) / img_size
              img[i, j] = values[t] * np.exp(-dist * 5)
        sequence.append(img)

      self.data.append(np.stack(sequence))
      self.labels.append(values[-1] * 10 + np.random.normal(0, 0.5))  # 回归目标

    self.data = np.array(self.data)
    self.labels = np.array(self.labels)

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    # 添加通道维度 (1通道灰度图像)
    img_sequence = torch.FloatTensor(self.data[idx]).unsqueeze(1)  # (T,1,H,W)
    label = torch.FloatTensor([self.labels[idx]])
    return img_sequence, label


# 2. 测试函数
def test_model(model, test_loader, device):
  model.eval()
  test_loss = 0
  criterion = nn.MSELoss()
  predictions = []
  true_values = []

  with torch.no_grad():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      predictions.extend(outputs.cpu().numpy().flatten())
      true_values.extend(targets.cpu().numpy().flatten())

  avg_loss = test_loss / len(test_loader)
  print(f"Test Loss: {avg_loss:.4f}")

  # 绘制预测结果对比图
  plt.figure(figsize=(10, 5))
  plt.scatter(true_values, predictions, alpha=0.5)
  plt.plot(
    [min(true_values), max(true_values)], [min(true_values), max(true_values)], "r--"
  )
  plt.xlabel("True Values")
  plt.ylabel("Predictions")
  plt.title("True vs Predicted Values")
  plt.show()

  return avg_loss


# 3. 训练函数
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device="cpu"):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  best_val_loss = float("inf")
  train_losses = []
  val_losses = []

  for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
      inputs = inputs.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
      for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        val_loss += criterion(outputs, targets).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(
      f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
    )

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save(model.state_dict(), "best_model.pth")

  # 绘制训练曲线
  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label="Training Loss")
  plt.plot(val_losses, label="Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.title("Training and Validation Loss")
  plt.show()

  return model


# 4. 主程序
def main():
  # 参数设置
  img_size = 32
  time_steps = 10
  batch_size = 32
  epochs = 30
  lr = 0.001

  # 设备设置
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # 创建数据集
  dataset = SyntheticTimeSeriesImages(
    num_samples=2000, img_size=img_size, time_steps=time_steps
  )

  # 划分训练集、验证集、测试集
  train_size = int(0.7 * len(dataset))
  val_size = int(0.15 * len(dataset))
  test_size = len(dataset) - train_size - val_size

  train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
  )

  # 创建数据加载器
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size)
  test_loader = DataLoader(test_dataset, batch_size=batch_size)

  # 初始化模型
  model = CnnLstmRegressor(
    img_size=img_size, time_steps=time_steps, num_channels=1, lstm_units=64, fc_units=64
  ).to(device)

  # 打印模型结构
  print(model)

  # 训练模型
  trained_model = train_model(
    model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
  )

  # 测试模型
  print("\nTesting best model...")
  best_model = CnnLstmRegressor(
    img_size=img_size, time_steps=time_steps, num_channels=1, lstm_units=64, fc_units=64
  ).to(device)
  best_model.load_state_dict(torch.load("best_model.pth"))
  test_loss = test_model(best_model, test_loader, device)

  # 示例预测
  sample_data, sample_label = test_dataset[0]
  with torch.no_grad():
    prediction = best_model(sample_data.unsqueeze(0).to(device))
  print("\nSample Prediction:")
  print(f"True value: {sample_label.item():.4f}")
  print(f"Predicted value: {prediction.item():.4f}")


if __name__ == "__main__":
  main()
