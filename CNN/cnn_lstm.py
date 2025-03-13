import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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

    self.lstm = nn.LSTM(
      input_size=64 * (img_size // 4) * (img_size // 4),
      hidden_size=lstm_units,
      batch_first=True,
    )

    self.fc1 = nn.Linear(lstm_units, 64)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(64, 1)

  def forward(self, x):
    batch_size, time_steps, C, H, W = x.size()
    x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
    x = self.cnn(x)
    x = x.view(-1, self.time_steps, 64 * (x.shape[2]) * (x.shape[3]))
    x, _ = self.lstm(x)
    x = self.dropout(F.relu(self.fc1(x[:, -1, :])))
    return self.fc1(x)
