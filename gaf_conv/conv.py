import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import data.get_data as gd
import data.get_feature as gf

def resample(series, new_length):
  x_original = np.linspace(0,1, len(series))
  x_new = np.linspace((0, 1, new_length))
  f = interp1d(x_original, series, kind='linear')
  return f(x_new)

# 将每个充放电周期的时间序列归一化到[-1,1]区间，以满足GAF对输入的要求
# scaler = MinMaxScaler(feature_range=(-1,1))
# X_normalized = scaler.fit_transform()   # 原始数据

def gaf_show(norma_value,image_size):
  gaf = GramianAngularField(method="summation", image_size=image_size)
  gaf_image = gaf.fit_transform(norma_value.reshape(-1,1))
  return gaf_image

# def draw_plot(g_img):
def conv_gaf_image():
  dfs = gf.dfs
  df_keys = gf.df_keys
  for df_key in df_keys:
    Caps = gf.read_dfs_by_cycle_toCap(df_key, dfs)
    output_path = r"F:\New\Coding\GAF_VIT\images"+f"\\{df_key}-images\\"
    if not os.path.exists(output_path):
      try:
        os.makedirs(output_path, exist_ok=True)
        print(f"创建图像输出目录{output_path} 成功")
      except OSError:
        print(f"创建图像输出目录{output_path} 失败:{OSError}")


    length = len(Caps)

    for i in range(length):
      caps = Caps[i]
      print(caps)

      if len(caps) < 124:
        caps_last = caps[-1]
        for _ in range(124-len(caps)):
          caps = list(caps)
          caps.append(caps_last)

        caps = np.array(caps)

      print("this is len: ",len(caps))

      scaler = MinMaxScaler(feature_range=(-1,1))
      caps_normalized = scaler.fit_transform(caps.reshape(-1, 1)).flatten()
      print("this caps normalized: ", caps_normalized)

      image_size = len(caps)
      gaf = GramianAngularField(
        image_size= image_size,
        method = "summation",
        sample_range=(-1,1)
      )

      gaf_images = gaf.fit_transform(caps_normalized.reshape(1,-1))

      plt.figure(figsize=(5,5))
      plt.imshow(gaf_images[0], cmap="viridis", origin="lower")
      plt.xticks([])
      plt.yticks([])
      plt.axis("off")
      plt.tight_layout()
      plt.savefig(output_path+f"{df_key}"+f"-{i}.png")




