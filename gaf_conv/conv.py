import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import data.get_feature as gf

def resample(series, new_length):
  x_original = np.linspace(0,1, len(series))
  x_new = np.linspace(0, 1, new_length)
  f = interp1d(x_original, series, kind='linear')
  return f(x_new)


def conv_gaf_image():
  dfs = gf.dfs
  df_keys = gf.df_keys
  for df_key in df_keys:
    Caps = gf.read_dfs_by_cycle_toCap(df_key, dfs)
    output_path = r"F:\New\Coding\GAF_VIT\images3"+f"\\{df_key}-images\\"
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

      caps_resampled = np.array(resample(caps,128))

      print("this is len: ",len(caps_resampled))

      scaler = MinMaxScaler(feature_range=(-1,1))
      caps_normalized = scaler.fit_transform(caps.reshape(-1, 1)).flatten()
      print("this caps normalized: ", caps_normalized)

      image_size = len(caps)
      gaf_data = GramianAngularField(
        image_size= image_size,
        method = "summation",
        sample_range=(-1,1)
      )

      gaf_images = gaf_data.fit_transform(caps_normalized.reshape(1, -1))

      plt.figure(figsize=(5,5))
      plt.imshow(gaf_images[0], cmap="viridis", origin="lower")
      plt.xticks([])
      plt.yticks([])
      plt.axis("off")
      plt.tight_layout()
      plt.savefig(output_path+f"{df_key}"+f"-{i}.png", bbox_inches ="tight", pad_inches=0)


def xj_conv_gaf_image():
  all_battery_data = gf.get_xj_batch_data("Batch-1")
  for battery_idx, battery_data in all_battery_data.items():
    output_path = r"D:\1New\Coding\GAF_VIT\xj_images" +f"\\{battery_idx}"
    if not os.path.exists(output_path):
      try:
        os.makedirs(output_path, exist_ok=True)
        print(f"创建图像输出目录{output_path} 成功")
      except OSError:
        print(f"创建图像输出目录{output_path} 失败:{OSError}")
        
    charge_data = gf.get_xj_battery_data(battery_data,1)
    
    for cycle_data in charge_data:
      caps =  cycle_data["capacity"]
      vols = charge_data["voltage"]
      curs = charge_data["current"]
      temps= charge_data["temperature"]
      i = cycle_data["cycle"]
      # 去除第一个测试cycle
      if i== 1:
        continue
      # print(f"{i},{caps}")
      
      scaler = MinMaxScaler(feature_range=(-1,1))
      caps_normalized = scaler.fit_transform(caps.reshape(-1, 1)).flatten()
      print("this caps normalized: ", caps_normalized)

      image_size = caps.size
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
      plt.savefig(output_path+f"\\{battery_idx}"+f"-{i}.png", bbox_inches ="tight", pad_inches=0)


def tj_conv_gaf_image(batch_name:str):
  all_battery_data = gf.get_tj_all_datas(batch_name)
  for battery_idx, battery_data in all_battery_data.items():
    output_path = r"D:\1New\Coding\GAF_VIT\tj_images\\" +f"{batch_name}"+"\\"+ f"\\{battery_idx}"
    if not os.path.exists(output_path):
      try:
        os.makedirs(output_path, exist_ok=True)
        print(f"创建图像输出目录{output_path} 成功")
      except OSError:
        print(f"创建图像输出目录{output_path} 失败:{OSError}")
    
    charge_data = gf.get_tj_battery_data(battery_data)
    
    for cycle_data in charge_data:
      cycle = cycle_data["cycle"]
      # caps = cycle_data["capacity"]
      vols = cycle_data["voltage"]
      curs = cycle_data["current"]
      rel_vols = cycle_data["rel_voltage"]
      ch_rel_vols = cycle_data["charge_rel_voltage"]
      dch_rel_vols = cycle_data["discharge_rel_voltage"]

      re_data = gf.resample(dch_rel_vols, 224)
      
      scaler = MinMaxScaler(feature_range=(-1, 1))
      caps_normalized = scaler.fit_transform(re_data.reshape(-1, 1)).flatten()
      print("this caps normalized: ", caps_normalized)
      
      image_size = re_data.size
      gaf = GramianAngularField(
        image_size=image_size,
        method="summation",
        sample_range=(-1, 1)
      )
      
      gaf_images = gaf.fit_transform(caps_normalized.reshape(1, -1))
      
      plt.figure(figsize=(5, 5))
      plt.imshow(gaf_images[0], cmap="viridis", origin="lower")
      plt.xticks([])
      plt.yticks([])
      plt.axis("off")
      plt.tight_layout()
      plt.savefig(output_path + f"\\{battery_idx}" + f"-{cycle}.png", bbox_inches="tight", pad_inches=0)

if __name__ == '__main__':
    # conv_gaf_image()
    # xj_conv_gaf_image()
    tj_conv_gaf_image("Dataset_1_NCA_battery")

