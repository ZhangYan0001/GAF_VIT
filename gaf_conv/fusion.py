"""
三通道融合，图像转换
"""
import os
import matplotlib.pyplot as plt
from data.xjbattery import Battery
from data.tjbattery import TJBattery
import numpy as np
from typing import Union
# from data.get_feature import get_xj_battery_data
import data.get_feature as gf
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
# import data.data_enhavence as de


def gaf(data):
  scaler = MinMaxScaler(feature_range=(0, 1))
  data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
  # print("data normalized: ", data_normalized)
  image_size = data.size

  gaf_data = GramianAngularField(
    image_size=image_size, method="summation", sample_range=(0, 1)
  )
  gaf_data = np.array(gaf_data.fit_transform(data_normalized.reshape(1, -1)))[0]
  gaf_data = (gaf_data + 1.0) / 2.0
  # print("the gaf_data:", gaf_data)

  # plt.imshow(gaf_data, cmap="gray")
  # plt.axis("off")
  # plt.show()
  # plt.close()

  return gaf_data


def fusion(data1, data2, data3):
  fusion_data = np.stack([data1, data2, data3], axis=-1)
  # print("the fusion_data: ", fusion_data)
  # print("the fusion_data shape: ", fusion_data.shape)
  return fusion_data


def three_channels_fusion(battery:Union[Battery, TJBattery], stage: int = 1, batch = ""):
  bdata_ = None
  fusion_datas = []
  print(battery.battery_name)
  output_path = r"D:\1New\Coding\GAF_VIT\fusion_images"+'\\'+batch+'\\'+f"{battery.battery_name}"
  if not os.path.exists(output_path):
      try:
        os.makedirs(output_path, exist_ok=True)
        print(f"创建图像输出目录{output_path} 成功")
      except OSError:
        print(f"创建图像输出目录{output_path} 失败:{OSError}")
      
  if isinstance(battery, Battery):
    bdata_ = gf.get_xj_battery_data(battery, stage)
  elif isinstance(battery, TJBattery):
    bdata_ = gf.get_tj_battery_data(battery)
  else:
    print("the battery is unknown")
    return None

  for cycle_data in bdata_:
    # i = cycle_data["cycle"]-1
    vol = cycle_data["voltage"]
    cur = cycle_data["current"]
    rel_vol = cycle_data["charge_rel_voltage"]
    
    if len(rel_vol) < 2:
      continue
    
    if isinstance(battery, TJBattery):
      vol, cur, rel_vol = gf.resample(vol,224),gf.resample(cur, 224), gf.resample(rel_vol,224)
    
    vol_, cur_, rel_vol_= gaf(vol), gaf(cur), gaf(rel_vol)
    fusion_data = fusion(vol_, cur_, rel_vol_)
    # augmentor = de.SimSiamGAF3Augmentor()
    # augmented_fusion_data = augmentor(fusion_data)
    # augmented_fusion_data = augmented_fusion_data.permute(1, 2, 0)
    # fusion_datas.append(augmented_fusion_data)
    fusion_datas.append(fusion_data)
    
    
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(fusion_data)
    # plt.show()
    plt.savefig(output_path + f"\\{battery.battery_name}" + f"-{cycle_data['cycle']}.png",bbox_inches ="tight", pad_inches=0)
    plt.close()
    
  return fusion_datas


"""
the test code
"""

# if __name__ == "__main__":
if __name__ == "__main__":
  for batch in gf.tj_batches_arr:
    tj_datas_1 = gf.get_tj_all_datas(batch)
    for _ , battery in tj_datas_1.items():
      three_channels_fusion(battery,batch=batch)
      # break
    break

  for batch in gf.batches_arr:
    xj_datas_1 = gf.get_xj_batch_data(batch)
    for _, battery in xj_datas_1.items():
      three_channels_fusion(battery=battery, batch = batch)
      # break
    break
  
  #   # print(
  #   #   "this is files path ",
  #   #   read_xj_data_files(xj_data_files_path, batches_arr)
  #   #
  # datas = gf.get_xj_data_file(
  #   gf.read_xj_data_files(gf.xj_data_files_path, gf.batches_arr), "Batch-1"
  # )
  #   # charge_datas = get_xj_battery_charge_data(batch1_all_battery_data['1'])
  #   # print("this is the 1 battery ", charge_datas)
  #   # print(batch1_all_battery_data['1'].get_one_cycle_description(1))
  # battery1 = datas["2C_battery-1"]
  # fusion_datas = three_channels_fusion(battery1, 1)
  # bdata_ = gf.get_xj_battery_data(battery1, 1)
  # fusion_datas = []
  # for cycle in bdata_:
  #   i = cycle["cycle"]-1
  #   # cap_ = bdata_[i]["capacity"]
  #   vol_ = bdata_[i]["voltage"]
  #   cur_ = bdata_[i]["current"]
  #   temp_ = bdata_[i]["temperature"]
  #   # cap_ = gaf(cap_)
  #   vol_, cur_, temp_ = gaf(vol_), gaf(cur_), gaf(temp_)
  #   fusion_data = fusion(vol_, cur_, temp_)
  #   print("the fusion data", fusion_data.shape)
  #   plt.imshow(fusion_data, cmap="gray")
  #   plt.title("128x128x3 image")
  #   plt.show()
  #   plt.close()
  #   fusion_datas.append(fusion_data)