import torch

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
  print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")

  # 显示每个 CUDA 设备的名称
  for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
  print("CUDA is not available")
