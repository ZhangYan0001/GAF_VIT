from ViT.pvit_model import PyramidVisionTransformerV2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

gaf_augmentation = transforms.Compose(
  [
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
  ]
)


class SSLHead(nn.Module):
  def __init__(self, embed_dim=512, proj_dim=128):
    super().__init__()
    self.projection = nn.Sequential(
      nn.Linear(embed_dim, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, proj_dim)
    )
  
  def forward(self, x):
    return self.projection(x)


class SSL_PVTv2(PyramidVisionTransformerV2):
  def __init__(self, **kwargs):
    super().__init__(in_chans=1, num_classes=0)
    # self.proj_head = SSLHead(embed_dims[-1])
  
  def forward(self, x):
    features = super().forward_features(x)
    return self.proj_head(features)


# 1. 时序数据转GAF图像的工具函数
def data_to_gasf(data, min_val=None, max_val=None):
  """
  将时序数据转换为GASF (Gramian Angular Summation Field)

  参数:
      data: 一维时间序列数据
      min_val, max_val: 可选的归一化范围

  返回:
      GASF矩阵
  """
  # 归一化到[-1, 1]之间
  scaler = MinMaxScaler(feature_range=(-1, 1))
  if min_val is not None and max_val is not None:
    # 使用提供的范围进行手动缩放
    normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
  else:
    # 使用sklearn的MinMaxScaler
    normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
  
  # 转换为极坐标：半径和角度
  phi = np.arccos(normalized_data)
  
  # 计算GASF矩阵
  gasf = np.cos(phi.reshape(-1, 1) + phi.reshape(1, -1))
  
  return gasf


def data_to_gadf(data, min_val=None, max_val=None):
  """
  将时序数据转换为GADF (Gramian Angular Difference Field)

  参数:
      data: 一维时间序列数据
      min_val, max_val: 可选的归一化范围

  返回:
      GADF矩阵
  """
  # 归一化到[-1, 1]之间
  scaler = MinMaxScaler(feature_range=(-1, 1))
  if min_val is not None and max_val is not None:
    # 使用提供的范围进行手动缩放
    normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
  else:
    # 使用sklearn的MinMaxScaler
    normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
  
  # 转换为极坐标：半径和角度
  phi = np.arccos(normalized_data)
  
  # 计算GADF矩阵
  gadf = np.sin(phi.reshape(-1, 1) - phi.reshape(1, -1))
  
  return gadf


# 2. 自定义GAF图像数据集
class GAFDataset(Dataset):
  def __init__(self, time_series_data, transform=None, window_size=None, stride=1):
    """
    用于GAF图像的自监督学习数据集

    参数:
        time_series_data: 2D数组，形状为[序列数，序列长度]
        transform: PyTorch转换函数
        window_size: 时间窗口大小，如果为None则使用完整序列
        stride: 滑动窗口的步长
    """
    self.data = time_series_data
    self.transform = transform
    self.window_size = window_size
    self.stride = stride
    
    # 如果指定了窗口大小，则创建滑动窗口索引
    self.windows = []
    if window_size is not None:
      for i in range(len(time_series_data)):
        for j in range(0, len(time_series_data[i]) - window_size + 1, stride):
          self.windows.append((i, j))
  
  def __len__(self):
    if self.window_size is not None:
      return len(self.windows)
    return len(self.data)
  
  def __getitem__(self, idx):
    if self.window_size is not None:
      # 使用滑动窗口
      seq_idx, start_idx = self.windows[idx]
      time_series = self.data[seq_idx][start_idx:start_idx + self.window_size]
    else:
      # 使用完整序列
      time_series = self.data[idx]
    
    # 转换为GAF图像
    gasf = data_to_gasf(time_series)
    gadf = data_to_gadf(time_series)
    
    # 将GAF图像堆叠为2通道图像
    gaf_image = np.stack([gasf, gadf], axis=0)
    gaf_tensor = torch.FloatTensor(gaf_image)
    
    # 应用转换
    if self.transform:
      gaf_tensor = self.transform(gaf_tensor)
    
    return gaf_tensor


# 3. 掩码重建自监督任务
class MaskedGAFReconstruction(nn.Module):
  def __init__(self):
    super(MaskedGAFReconstruction, self).__init__()
    
    # 编码器 (使用ResNet的前几层)
    resnet = models.resnet18(pretrained=False)
    self.encoder = nn.Sequential(
      nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 修改第一层接受2通道输入
      resnet.bn1,
      resnet.relu,
      resnet.maxpool,
      resnet.layer1,
      resnet.layer2
    )
    
    # 解码器 (转置卷积重建)
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),
      nn.Tanh()  # 输出范围[-1, 1]，匹配GAF图像
    )
  
  def forward(self, x, mask=None):
    # 如果提供了掩码，则将输入的对应区域置零
    if mask is not None:
      x = x * (1 - mask)
    
    # 编码并解码
    features = self.encoder(x)
    reconstructed = self.decoder(features)
    
    return reconstructed


# 4. 时间片段预测任务
def create_time_masks(batch_size, channels, size, num_masks=1, mask_size=0.2):
  """创建随机时间掩码"""
  masks = torch.zeros(batch_size, channels, size, size)
  mask_length = int(size * mask_size)
  
  for i in range(batch_size):
    for _ in range(num_masks):
      start = np.random.randint(0, size - mask_length)
      # 掩盖特定的时间段 (对角线区域)
      for j in range(start, start + mask_length):
        masks[i, :, j, :] = 1.0
        masks[i, :, :, j] = 1.0
  
  return masks


# 5. 对比学习模型
class GAFContrastiveLearning(nn.Module):
  def __init__(self, feature_dim=128):
    super(GAFContrastiveLearning, self).__init__()
    
    # 使用轻量级的ResNet作为骨干网络
    resnet = models.resnet18(pretrained=False)
    # 修改第一层以接受2通道输入
    resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 编码器 (ResNet除去全连接层)
    self.encoder = nn.Sequential(*list(resnet.children())[:-1])
    
    # 投影头 (MLP投影到低维空间)
    self.projector = nn.Sequential(
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, feature_dim)
    )
  
  def forward(self, x):
    # 特征提取
    features = self.encoder(x)
    features = torch.flatten(features, start_dim=1)
    
    # 投影到低维表示空间
    z = self.projector(features)
    # 归一化特征向量
    z = nn.functional.normalize(z, dim=1)
    
    return z


# 6. 时间序列对比损失函数
class GAFInfoNCELoss(nn.Module):
  def __init__(self, temperature=0.5):
    super(GAFInfoNCELoss, self).__init__()
    self.temperature = temperature
    self.criterion = nn.CrossEntropyLoss()
  
  def forward(self, features_1, features_2):
    # 计算批大小
    batch_size = features_1.shape[0]
    
    # 创建标签：对于每个样本，正样本是另一个视图的对应表示
    labels = torch.arange(batch_size, device=features_1.device)
    
    # 计算相似度矩阵
    features_1_norm = nn.functional.normalize(features_1, dim=1)
    features_2_norm = nn.functional.normalize(features_2, dim=1)
    
    similarity_matrix = torch.matmul(features_1_norm, features_2_norm.T) / self.temperature
    
    # 计算对比损失
    loss_1 = self.criterion(similarity_matrix, labels)
    loss_2 = self.criterion(similarity_matrix.T, labels)
    
    # 总损失是双向损失的平均值
    loss = (loss_1 + loss_2) / 2.0
    
    return loss


# 7. 数据增强函数
class GAFDataAugmentation:
  def __init__(self, jitter_scale=0.1, time_warp_scale=0.2):
    self.jitter_scale = jitter_scale
    self.time_warp_scale = time_warp_scale
  
  def __call__(self, time_series):
    """
    对时间序列应用增强，然后转换为GAF图像

    参数:
        time_series: 一维numpy数组

    返回:
        两个增强后的GAF图像张量
    """
    # 应用抖动增强
    jittered = self._add_jitter(time_series.copy())
    
    # 应用时间扭曲增强
    warped = self._time_warp(time_series.copy())
    
    # 转换为GAF图像
    gasf_1 = data_to_gasf(jittered)
    gadf_1 = data_to_gadf(jittered)
    gaf_1 = torch.FloatTensor(np.stack([gasf_1, gadf_1], axis=0))
    
    gasf_2 = data_to_gasf(warped)
    gadf_2 = data_to_gadf(warped)
    gaf_2 = torch.FloatTensor(np.stack([gasf_2, gadf_2], axis=0))
    
    return gaf_1, gaf_2
  
  def _add_jitter(self, x):
    """添加随机噪声"""
    noise = np.random.normal(0, self.jitter_scale, x.shape)
    return x + noise
  
  def _time_warp(self, x):
    """应用时间扭曲（随机拉伸和压缩某些部分）"""
    length = len(x)
    num_knots = int(length / 10)  # 控制点数量
    knot_xs = np.arange(0, length, length // num_knots)
    knot_ys = np.arange(0, length, length // num_knots) * (
      1 + np.random.uniform(-self.time_warp_scale, self.time_warp_scale, size=len(knot_xs)))
    
    # 线性插值
    warped_x = np.interp(np.arange(length), knot_xs, knot_ys).astype(int)
    warped_x = np.clip(warped_x, 0, length - 1)
    
    return x[warped_x]


# 8. 训练函数：掩码重建
def train_reconstruction(model, train_loader, epochs=100, lr=0.001, device="cuda"):
  """训练GAF图像掩码重建模型"""
  # 设置优化器和损失函数
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  
  model.train()
  for epoch in range(epochs):
    total_loss = 0.0
    
    for batch_idx, gaf_images in enumerate(train_loader):
      gaf_images = gaf_images.to(device)
      batch_size = gaf_images.shape[0]
      
      # 创建随机掩码
      masks = create_time_masks(
        batch_size=batch_size,
        channels=gaf_images.shape[1],
        size=gaf_images.shape[2]
      ).to(device)
      
      # 前向传播
      reconstructed = model(gaf_images, masks)
      
      # 只计算被掩盖区域的重建损失
      loss = criterion(reconstructed * masks, gaf_images * masks)
      
      # 反向传播和优化
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      
      if batch_idx % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.6f}")
  
  return model


# 9. 训练函数：对比学习
def train_contrastive(model, train_dataset, epochs=100, batch_size=64, lr=0.001, device="cuda"):
  """训练GAF图像对比学习模型"""
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  
  # 设置优化器和损失函数
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = GAFInfoNCELoss(temperature=0.1)
  
  model.train()
  for epoch in range(epochs):
    total_loss = 0.0
    
    for batch_idx, time_series_batch in enumerate(train_loader):
      # 对时间序列应用增强并转换为GAF
      views = []
      for time_series in time_series_batch:
        # 应用增强和GAF转换
        gasf_1 = data_to_gasf(time_series.numpy())
        gadf_1 = data_to_gadf(time_series.numpy())
        view_1 = torch.FloatTensor(np.stack([gasf_1, gadf_1], axis=0))
        
        # 对时间序列应用随机增强
        augmenter = GAFDataAugmentation()
        augmented = augmenter._add_jitter(time_series.numpy())
        
        gasf_2 = data_to_gasf(augmented)
        gadf_2 = data_to_gadf(augmented)
        view_2 = torch.FloatTensor(np.stack([gasf_2, gadf_2], axis=0))
        
        views.append((view_1, view_2))
      
      # 分离两个视图
      view_1_batch = torch.stack([v[0] for v in views]).to(device)
      view_2_batch = torch.stack([v[1] for v in views]).to(device)
      
      # 前向传播
      z_1 = model(view_1_batch)
      z_2 = model(view_2_batch)
      
      # 计算对比损失
      loss = criterion(z_1, z_2)
      
      # 反向传播和优化
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      
      if batch_idx % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.6f}")
  
  return model


# 10. 下游任务评估
class GAFClassifier(nn.Module):
  def __init__(self, pretrained_encoder, num_classes):
    super(GAFClassifier, self).__init__()
    self.encoder = pretrained_encoder
    # 冻结编码器参数
    for param in self.encoder.parameters():
      param.requires_grad = False
    self.classifier = nn.Linear(512, num_classes)
  
  def forward(self, x):
    with torch.no_grad():
      features = self.encoder(x)
      features = torch.flatten(features, start_dim=1)
    return self.classifier(features)


# 11. 时序分类评估函数
def evaluate_classifier(model, test_loader, device="cuda"):
  """评估模型在测试数据上的性能"""
  model.eval()
  correct = 0
  total = 0
  
  with torch.no_grad():
    for gaf_images, labels in test_loader:
      gaf_images, labels = gaf_images.to(device), labels.to(device)
      outputs = model(gaf_images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  accuracy = 100 * correct / total
  print(f"Test Accuracy: {accuracy:.2f}%")
  
  return accuracy


# 12. 可视化GAF图像
def visualize_gaf(time_series, title="Time Series GAF Representation"):
  """可视化原始时间序列和对应的GAF图像"""
  plt.figure(figsize=(15, 5))
  
  # 绘制原始时间序列
  plt.subplot(1, 3, 1)
  plt.plot(time_series)
  plt.title("Original Time Series")
  plt.grid(True)
  
  # 计算并绘制GASF
  gasf = data_to_gasf(time_series)
  plt.subplot(1, 3, 2)
  plt.imshow(gasf, cmap="viridis")
  plt.colorbar()
  plt.title("GASF")
  
  # 计算并绘制GADF
  gadf = data_to_gadf(time_series)
  plt.subplot(1, 3, 3)
  plt.imshow(gadf, cmap="viridis")
  plt.colorbar()
  plt.title("GADF")
  
  plt.suptitle(title)
  plt.tight_layout()
  plt.show()


# 13. 主函数示例
def main(time_series_data, labels=None):
  """
  完整的GAF自监督学习流程

  参数:
      time_series_data: 形状为[序列数, 序列长度]的numpy数组
      labels: 可选，时间序列的类别标签
  """
  # 设置设备
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # 创建GAF数据集
  dataset = GAFDataset(time_series_data, window_size=128, stride=16)
  
  # 数据加载器
  train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  # 1. 掩码重建训练
  print("Training GAF reconstruction model...")
  recon_model = MaskedGAFReconstruction().to(device)
  recon_model = train_reconstruction(recon_model, train_loader, epochs=50, device=device)
  
  # 保存重建模型
  torch.save(recon_model.state_dict(), "gaf_reconstruction_model.pth")
  
  # 2. 对比学习训练
  print("Training GAF contrastive learning model...")
  contrastive_model = GAFContrastiveLearning().to(device)
  contrastive_model = train_contrastive(contrastive_model, dataset, epochs=50, device=device)
  
  # 保存对比学习模型
  torch.save(contrastive_model.state_dict(), "gaf_contrastive_model.pth")
  
  # 如果有标签，可以评估下游任务性能
  if labels is not None:
    # 创建带标签的数据集
    test_dataset = GAFDataset(time_series_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建分类器并评估性能
    classifier = GAFClassifier(contrastive_model.encoder, num_classes=len(np.unique(labels))).to(device)
    
    # 训练分类器
    classifier_optimizer = optim.Adam(classifier.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Training classifier for evaluation...")
    for epoch in range(30):
      classifier.train()
      for gaf_images, batch_labels in test_loader:
        gaf_images, batch_labels = gaf_images.to(device), batch_labels.to(device)
        
        outputs = classifier(gaf_images)
        loss = criterion(outputs, batch_labels)
        
        classifier_optimizer.zero_grad()
        loss.backward()
        classifier_optimizer.step()
      
      # 每10个epoch评估一次
      if epoch % 10 == 9:
        evaluate_classifier(classifier, test_loader, device=device)


if __name__ == "__main__":
  main()
