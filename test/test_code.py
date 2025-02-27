import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
import data.get_feature as gf

# 假设输入数据
Caps = gf.read_dfs_by_cycle_toCap("XQ-11",gf.dfs)
# times = np.linspace(0, len(Caps[1]), len(Caps[1]))  # 时间点，假设有100个时间点
Times = gf.read_dfs_by_cycle_toTime("XQ-11", gf.dfs)
# caps = 2.0 + 0.5 * np.sin(times / 10) + np.random.normal(0, 0.05, len(times))  # 容量数据，带噪声
# caps = Caps[0]
# times = np.linspace(0, len(Times[0]), len(Times[0]))
length = len(Caps)
output_path = r"F:\New\Coding\GAF_VIT\images\XQ-11-images"

for i in range(length):
    caps = Caps[i]
    times = Times[i]
    print(caps)

    # 差值统一大小
    if len(caps) < 124:
        caps_last = caps[-1]
        for _ in range(124-len(caps)):
            caps = list(caps)
            caps.append(caps_last)
        caps = np.array(caps)

    print("this is len :",len(caps))

    # 1. 数据归一化到 [-1, 1] 区间（GAF的要求）
    scaler = MinMaxScaler(feature_range=(-1, 1))
    caps_normalized = scaler.fit_transform(caps.reshape(-1, 1)).flatten()
    print("this caps normalized :", caps_normalized)

    # 2. 创建 GAF 转换器
    image_size = len(caps)  # 图像尺寸设置为时间序列长度
    gaf = GramianAngularField(
        image_size=image_size,  # 图像尺寸
        method='summation',  # 使用 GASF（Gramian Angular Summation Field）
        sample_range=(-1, 1)  # 数据范围
    )

    # 3. 转换为 GAF 图像
    gaf_images = gaf.fit_transform(caps_normalized.reshape(1, -1))  # 输入需要是二维数组

    # 4. 可视化结果
    plt.figure(figsize=(12, 5))

    # 原始容量时序数据
    # plt.subplot(1, 2, 1)
    # plt.plot(times, caps, color='blue', label='Capacity')
    # plt.title('Capacity vs Time')
    # plt.xlabel('Time')
    # plt.ylabel('Capacity (Ah)')
    # plt.legend()

    # GAF 图像
    # plt.subplot(1, 2, 2)
    plt.imshow(gaf_images[0], cmap='viridis', origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    # plt.title('Gramian Angular Summation Field (GASF)')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Time Step')
    # plt.ylabel('Time Step')

    plt.tight_layout()
    # plt.savefig(output_path+f"\\XQ-11-{i}.png")
    plt.show()

# print(caps)
#
# # 1. 数据归一化到 [-1, 1] 区间（GAF的要求）
# scaler = MinMaxScaler(feature_range=(-1, 1))
# caps_normalized = scaler.fit_transform(caps.reshape(-1, 1)).flatten()
# print("this caps normalized :",caps_normalized)
#
# # 2. 创建 GAF 转换器
# image_size = len(caps)  # 图像尺寸设置为时间序列长度
# gaf = GramianAngularField(
#     image_size=image_size,  # 图像尺寸
#     method='summation',     # 使用 GASF（Gramian Angular Summation Field）
#     sample_range=(-1, 1)    # 数据范围
# )
#
# # 3. 转换为 GAF 图像
# gaf_images = gaf.fit_transform(caps_normalized.reshape(1, -1))  # 输入需要是二维数组
#
# # 4. 可视化结果
# plt.figure(figsize=(12, 5))
#
# # 原始容量时序数据
# plt.subplot(1, 2, 1)
# plt.plot(Times[0], caps, color='blue', label='Capacity')
# plt.title('Capacity vs Time')
# plt.xlabel('Time')
# plt.ylabel('Capacity (Ah)')
# plt.legend()
#
# # GAF 图像
# plt.subplot(1, 2, 2)
# plt.imshow(gaf_images[0], cmap='viridis', origin='lower')
# plt.title('Gramian Angular Summation Field (GASF)')
# plt.colorbar(label='Magnitude')
# plt.xlabel('Time Step')
# plt.ylabel('Time Step')
#
# plt.tight_layout()
# plt.savefig(output_path+f"\\XQ-11-{1}.png")
# plt.show()