import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import data.get_dataset as gds

# 测试配置
TEST_INDEX = 0          # 要测试的样本索引
BATCH_SIZE = 4          # 测试批次大小
SHOW_SAMPLE = True      # 是否可视化样本

dataset = gds.BatteryDataset(
  img_dir=gds.image_path,
  img_keys=gds.image_keys,
  transform=gds.get_transform()
)
# 测试1：基础检查
print(f"数据集总样本数: {len(dataset)}")
print(f"数据集总标签数: {len(dataset.labels)}")
print(f"样本 {TEST_INDEX} 的路径: {dataset.path_data[TEST_INDEX]}")
print(f"样本 {TEST_INDEX} 的标签: {dataset.labels[TEST_INDEX]}")

# 测试2：获取单个样本
try:
    sample_img, sample_label = dataset[TEST_INDEX]
    print("\n单个样本测试通过")
    print(f"图像类型: {type(sample_img)} | 形状: {sample_img.shape}")
    print(f"标签类型: {type(sample_label)} | 值: {sample_label}")
except Exception as e:
    print(f"\n单个样本加载失败: {str(e)}")

# 测试3：DataLoader批量加载
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
try:
    batch = next(iter(dataloader))
    imgs, labels = batch
    print("\n批量加载测试通过")
    print(f"批次图像形状: {imgs.shape}")  # 应显示为 [BATCH_SIZE, C, H, W]
    print(f"批次标签形状: {labels.shape}")
except Exception as e:
    print(f"\n批量加载失败: {str(e)}")