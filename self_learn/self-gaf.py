import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 1. 用于生成GAF图像的工具类
class GAFGenerator:
  """
  将时间序列数据转换为Gramian Angular Field图像
  """

  def __init__(self, method="summation", image_size=224):
    """
    初始化GAF生成器

    参数:
    - method: 'summation'表示GASF, 'difference'表示GADF
    - image_size: 输出图像的大小
    """
    self.method = method
    self.image_size = image_size
    self.gaf = GramianAngularField(image_size=image_size, method=method)
    self.scaler = MinMaxScaler(feature_range=(0, 1))

  def _interpolate(self, series, new_length):
    original_length = series.shape[0]
    if original_length == new_length:
      return series
    x_orig = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)
    f = interp1d(x_orig, series, kind="linear")
    return f(x_new)

  def transform(self, time_series):
    """
    将时间序列转换为GAF图像

    参数:
    - time_series: 形状为(n_samples, n_timestamps)的时间序列数据

    返回:
    - GAF图像，形状为(n_samples, image_size, image_size)
    """
    # 确保输入是2D数组
    if time_series.ndim == 1:
      time_series = time_series.reshape(1, -1)

    n_samples = time_series.shape[0]
    gaf_images = []

    interpolated = np.apply_along_axis(
      lambda x: self._interpolate(x, self.image_size), 1, time_series
    )
    
    normalized = self.scaler.fit_transform(interpolated)
    
    for sample in normalized:
      img = self.gaf.fit_transform(sample.reshape(1, -1))
      gaf_images.append(img[0])


    return np.array(gaf_images)

    # # 归一化到[0,1]区间
    # time_series_norm = self.scaler.fit_transform(time_series).reshape(-1,1)
    #
    # # 生成GAF图像
    # gaf_images = self.gaf.fit_transform(time_series_norm.reshape(1,-1))
    #
    # return gaf_images

  def visualize(self, time_series, index=0):
    """
    可视化时间序列和对应的GAF图像
    """
    # 生成GAF图像
    gaf_images = self.transform(time_series)

    # 创建绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制时间序列
    if time_series.ndim == 1:
      ax1.plot(time_series)
    else:
      ax1.plot(time_series[index])
    ax1.set_title("Time Series")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")

    # 绘制GAF图像
    im = ax2.imshow(gaf_images[index], cmap="viridis")
    ax2.set_title(f"Gramian Angular {self.method.capitalize()} Field")
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.show()


# 2. 自定义数据集类
class GAFDataset(Dataset):
  def __init__(self, time_series, transform=None, gaf_generator=None):
    """
    GAF图像数据集

    参数:
    - time_series: 形状为(n_samples, n_timestamps)的时间序列数据
    - transform: 图像增强转换
    - gaf_generator: GAF生成器实例
    """
    self.time_series = time_series
    self.transform = transform

    if gaf_generator is None:
      self.gaf_generator = GAFGenerator()
    else:
      self.gaf_generator = gaf_generator

    # 生成所有GAF图像
    self.gaf_images = self.gaf_generator.transform(time_series)

  def __len__(self):
    return len(self.time_series)

  def __getitem__(self, idx):
    # 获取GAF图像
    image = self.gaf_images[idx]

    # 将单通道图像扩展为3通道
    image = np.stack([image] * 3, axis=0)
    image = torch.FloatTensor(image)

    # 应用图像增强
    if self.transform:
      image1 = self.transform(image)
      image2 = self.transform(image)
      return (image1, image2)
    else:
      return image


# 3. 为时序GAF图像定制的自监督对比学习模型
class GAFSimCLR(nn.Module):
  def __init__(self, feature_dim=128, use_attention=True):
    super(GAFSimCLR, self).__init__()

    # 使用ResNet18作为骨干网络
    resnet = models.resnet18(pretrained=False)
    self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    # 特征维度
    self.feature_dim = feature_dim
    self.use_attention = use_attention

    # 如果使用注意力机制来捕获时序关系
    if use_attention:
      self.attention = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(512, 1, kernel_size=1),
        nn.Sigmoid(),
      )

    # 投影头
    self.projector = nn.Sequential(
      nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.feature_dim)
    )

  def forward(self, x):
    features = self.encoder(x)

    if self.use_attention:
      # 应用通道注意力
      attention_weights = self.attention(features)
      features = features * attention_weights

    features = torch.flatten(features, start_dim=1)
    projections = self.projector(features)

    # 归一化表示向量
    projections = nn.functional.normalize(projections, dim=1)
    return projections


# 4. InfoNCE损失函数
class InfoNCELoss(nn.Module):
  def __init__(self, temperature=0.5):
    super(InfoNCELoss, self).__init__()
    self.temperature = temperature
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, features):
    # 分离正样本对
    batch_size = features.shape[0] // 2
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T) / self.temperature

    # 移除对角线上的自相似度
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    similarity_matrix = similarity_matrix[~mask].reshape(labels.shape[0], -1)
    labels = labels[~mask].reshape(labels.shape[0], -1)

    # 正负样本的相似度
    positives = similarity_matrix[labels.bool()].reshape(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].reshape(
      similarity_matrix.shape[0], -1
    )

    # 构建logits和标签
    logits = torch.cat([positives, negatives], dim=1)
    target = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    # 计算损失
    loss = self.criterion(logits, target)
    return loss


# 5. 时序一致性对比任务
class TemporalConsistencyLoss(nn.Module):
  """
  促使模型学习时间序列中的时序一致性
  """

  def __init__(self, temperature=0.5):
    super(TemporalConsistencyLoss, self).__init__()
    self.temperature = temperature

  def forward(self, z1, z2, temporal_distances):
    """
    基于时序距离的对比损失

    参数:
    - z1, z2: 表示向量批次
    - temporal_distances: 批次中样本间的时序距离
    """
    batch_size = z1.shape[0]

    # 计算余弦相似度
    sim_matrix = torch.mm(z1, z2.T) / self.temperature

    # 基于时序距离计算权重
    # 时序上更近的样本应该有更相似的表示
    temporal_weights = torch.exp(-temporal_distances / temporal_distances.max())

    # 加权的对比损失
    loss = -torch.log(
      torch.sum(torch.diag(sim_matrix) * temporal_weights)
      / torch.sum(sim_matrix * (1 - torch.eye(batch_size).to(z1.device)))
    )

    return loss


# 6. 趋势预测任务
class TrendPredictionHead(nn.Module):
  """
  预测时间序列的趋势变化作为自监督任务
  """

  def __init__(self, input_dim=512, hidden_dim=256, num_classes=3):
    """
    num_classes=3 表示 上升、平稳、下降
    """
    super(TrendPredictionHead, self).__init__()
    self.predictor = nn.Sequential(
      nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
    )

  def forward(self, x):
    return self.predictor(x)


# 7. 训练自监督模型的主函数
def train_gaf_simclr(
  model, train_loader, optimizer, criterion, epochs=100, device="cuda"
):
  model.train()
  for epoch in range(epochs):
    total_loss = 0.0
    for batch_idx, (images1, images2) in enumerate(train_loader):
      images1 = images1.to(device)
      images2 = images2.to(device)

      # 清零梯度
      optimizer.zero_grad()

      # 前向传播
      z1 = model(images1)
      z2 = model(images2)

      # 拼接所有表示
      representations = torch.cat([z1, z2], dim=0)

      # 计算对比损失
      loss = criterion(representations)

      # 反向传播和优化
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      if batch_idx % 20 == 0:
        print(
          f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.4f}")

  return model


# 8. 下游任务模型：时间序列分类
class TimeSeriesClassifier(nn.Module):
  def __init__(self, encoder, num_classes):
    super(TimeSeriesClassifier, self).__init__()
    self.encoder = encoder
    # 冻结编码器参数
    for param in self.encoder.parameters():
      param.requires_grad = False
    self.classifier = nn.Linear(512, num_classes)

  def forward(self, x):
    with torch.no_grad():
      features = self.encoder(x)
      features = torch.flatten(features, start_dim=1)
    return self.classifier(features)


# 9. 下游任务模型：时间序列预测
class TimeSeriesPredictor(nn.Module):
  def __init__(self, encoder, pred_horizon=1):
    super(TimeSeriesPredictor, self).__init__()
    self.encoder = encoder
    # 在微调设置中可能希望解冻编码器
    for param in self.encoder.parameters():
      param.requires_grad = True

    self.predictor = nn.Sequential(
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, pred_horizon),
    )

  def forward(self, x):
    features = self.encoder(x)
    features = torch.flatten(features, start_dim=1)
    return self.predictor(features)


# 10. 主函数：完整的训练和评估流程
def main():
  # 设置设备
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 超参数
  batch_size = 64
  feature_dim = 128
  temperature = 0.1
  epochs = 100
  lr = 0.0003

  # 示例：生成或加载时间序列数据
  # 这里假设我们已经有了时间序列数据
  # time_series = np.random.randn(1000, 200)  # 1000个样本，每个200个时间点

  # 从文件加载时间序列数据
  # time_series = np.load('your_time_series_data.npy')

  # 为了演示，这里生成随机时间序列
  num_samples = 1000
  seq_length = 200
  time_series = np.zeros((num_samples, seq_length))

  # 生成具有趋势和季节性的时间序列
  for i in range(num_samples):
    # 添加趋势
    trend = np.linspace(0, np.random.uniform(-5, 5), seq_length)
    # 添加季节性
    seasonality = np.sin(np.linspace(0, np.random.randint(5, 15) * np.pi, seq_length))
    # 添加噪声
    noise = np.random.normal(0, 0.5, seq_length)
    # 组合
    time_series[i] = trend + seasonality + noise

  # 初始化GAF生成器
  gaf_generator = GAFGenerator(method="summation", image_size=224)

  # 创建数据增强
  transform = transforms.Compose(
    [
      transforms.RandomApply(
        [
          transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ],
        p=0.7,
      ),
      transforms.RandomApply(
        [
          transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ],
        p=0.5,
      ),
      transforms.RandomApply(
        [
          transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
        ],
        p=0.3,
      ),
    ]
  )

  # 创建数据集和数据加载器
  train_dataset = GAFDataset(
    time_series, transform=transform, gaf_generator=gaf_generator
  )
  train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
  )

  # 创建模型
  model = GAFSimCLR(feature_dim=feature_dim, use_attention=True).to(device)

  # 定义损失函数和优化器
  criterion = InfoNCELoss(temperature=temperature).to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

  # 训练模型
  model = train_gaf_simclr(
    model, train_loader, optimizer, criterion, epochs=epochs, device=device
  )

  # 保存预训练模型
  torch.save(model.state_dict(), "gaf_simclr_pretrained.pth")

  print("自监督预训练完成！")

  # 这里可以添加下游任务评估代码
  # 例如，时间序列分类或预测


if __name__ == "__main__":
  main()
