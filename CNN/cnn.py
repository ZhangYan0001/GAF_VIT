import torch
import torch.nn as nn


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


# 示例使用
if __name__ == "__main__":
  # 配置参数（与ViT实验保持一致）
  config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
    "weight_decay": 0.05,
    "epochs": 100
  }

  # 初始化模型
  model = CNNRegressor(
    input_size=128,
    in_channels=1,  # 匹配GAF单通道输入
    dropout_rate=0.3
  ).to(config["device"])

  # 验证输入输出维度
  test_input = torch.randn(4, 1, 128, 128).to(config["device"])
  output = model(test_input)
  print(f"输入尺寸: {test_input.shape} -> 输出尺寸: {output.shape}")
  # 应输出：输入尺寸: torch.Size([4, 1, 128, 128]) -> 输出尺寸: torch.Size([4]):
