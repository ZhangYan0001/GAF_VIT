import matplotlib.pyplot as plt
import numpy as np

# 模拟数据：x轴为测试步骤（0-15），颜色为循环次数（0-15）
steps = np.arange(16)
cycles = np.arange(16)
voltage = 4.19 - 0.0025 * steps + np.random.normal(0, 0.001, 16)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    steps,
    voltage,
    c=cycles,
    cmap='viridis',
    s=100,
    edgecolor='w'
)

# 坐标轴与颜色栏标注
plt.xlabel('Test Step', fontsize=12)
plt.ylabel('Voltage (V)', fontsize=12)
cbar = plt.colorbar(scatter)
cbar.set_label('Cycle Number', fontsize=12)

plt.title('Voltage vs. Test Step (Colored by Cycle)\nData from Tongji-NCM', pad=20)
plt.grid(True, alpha=0.3)
plt.show()