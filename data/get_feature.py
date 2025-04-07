import os.path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import interp1d

from data.xjbattery import Battery
from data.tjbattery import TJBattery
# from gaf_conv.fusion import fusion, gaf
import matplotlib.pyplot as plt

"""
  the XQ-_-25-1C 数据提取，包括时间，电压，容量，的数据按循环读取
"""


def read_file(file_path: str):
  if file_path.split(".")[-1] == "xlsx":
    df = pd.read_excel(file_path)
  else:
    df = pd.read_csv(file_path)
  return df


def get_df_data(file_path: str, file_names: []):
  dataframes: dict[Any, DataFrame] = {}
  for i in file_names:
    dataframes[i.split(".")[0][:5]] = read_file(file_path + "\\" + i)
  return dataframes


df_keys = [
  "XQ-11",
  "XQ-12",
  "XQ-14",
  "XQ-15",
  "XQ-16",
  "XQ-17",
  "XQ-18",
]
files_name = [
  "XQ-11-25-1C-pre.xlsx",
  "XQ-12-25-1C-pre.xlsx",
  "XQ-14-25-1C-pre.xlsx",
  "XQ-15-25-1C-pre.xlsx",
  "XQ-16-25-1C-pre.xlsx",
  "XQ-17-25-1C-pre.xlsx",
  "XQ-18-25-1C-pre.xlsx",
]

data_files_path = r"D:\1New\Coding\Datasets\data"
dfs = get_df_data(data_files_path, files_name)


def read_dfs_by_cycle_toCap(df_key, dfs_data: dict):
  df = dfs_data[df_key]
  cycles = list(set(df["循环"]))
  Caps = []
  for c in cycles:
    if c in [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]:
      continue
    df_lim = df[df["循环"] == c]
    Cap = np.array(list(df_lim["容量(Ah)"])).reshape(-1)
    Caps.append(Cap)

  return np.array(Caps, dtype=object)


def read_dfs_by_cycle_toTime(df_key, dfs_data: dict):
  df = dfs_data[df_key]
  cycles = list(set(df["循环"]))
  times = []
  for c in cycles:
    if c in [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]:
      continue
    df_lim = df[df["循环"] == c]
    time = np.array(list(df_lim["time"])).reshape(-1)
    times.append(time)
  return np.array(times, dtype=object)


def read_dfs_by_cycle_toSOH(df_key, dfs_data: dict):
  df = dfs_data[df_key]
  cycles = list(set(df["循环"]))
  SOHs_label = {}
  i = 0
  for c in cycles:
    if c in [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]:
      continue
    df_lim = df[df["循环"] == c]
    soh = np.array(list(df_lim["SoH"])).reshape(-1)
    sohs_avg = round(np.mean(soh), 6)
    SOHs_label[i] = sohs_avg
    i += 1
  return SOHs_label


def read_dfs_by_cycle_toVol(df_key, dfs_data: dict):
  df = dfs_data[df_key]
  cycles = list(set(df["循环"]))
  Vols = []
  for c in cycles:
    if c in [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]:
      continue
    df_lim = df[df["循环"] == c]
    time = np.array(list(df_lim["电压(V)"])).reshape(-1)
    Vols.append(time)
  return np.array(Vols, dtype=object)


def get_soh_labels(keys: [], label_flag: str, xj_batch="Batch-1", tj_batch="Dataset_1_NCA_battery"):
  SOH_Labels = {}
  if label_flag == "SE":
    for df_key in keys:
      soh_label = read_dfs_by_cycle_toSOH(df_key, dfs)
      SOH_Labels[df_key] = soh_label

  elif label_flag == "XJ":
    batch_data = get_xj_batch_data(xj_batch)
    for data_key in keys:
      soh_label = cal_xj_battery_soh_data(batch_data[data_key])
      SOH_Labels[data_key] = soh_label
  elif label_flag == "TJ":
    # todo 获取SOH
    batch_data = get_tj_all_datas(tj_batch)
    for key in keys:
      soh_label = cal_tj_soh_battery(batch_data[key])
      SOH_Labels[key] = soh_label
  else:
    print("当前数据集flag不存在，请重新输入")

  return SOH_Labels


"""
  XJTU 电池数据读取
"""

xj_data_files_path = r"D:\1New\Coding\Datasets\XJTU\Battery Dataset"
batches_arr = ["Batch-" + str(i) for i in range(1, 7)]


def read_xj_data_files(path: str, batches: list[str]):
  files_count = {}
  for batch in batches:
    current_path = os.path.join(path, batch)
    if not os.path.isdir(current_path):
      print(f"error: {current_path} is not exist.")
      break

    filenames = os.listdir(current_path)
    files_count[batch] = [
      os.path.join(current_path, filename) for filename in filenames
    ]

  return files_count


def get_xj_data_file(files_path: {str, list[str]}, batch: str):
  paths = files_path[batch]
  mat_datas = {}
  for path in paths:
    if os.path.isfile(path):
      # idx = path.split("\\")[-1].split(".")[0].split("-")[-1]
      battery = Battery(path)
      idx = battery.battery_name.split("\\")[-1]
      mat_datas[idx] = battery
  print(f"the {batch} and have {len(paths)} battery")
  return mat_datas


def get_xj_battery_data(battery: Battery, stage: int):
  # name = battery.battery_name
  cycle_life = battery.cycle_life
  data_ = []
  for i in range(1, cycle_life + 1):
    cap = battery.get_partial_value(i, 4, stage)
    vol = battery.get_partial_value(i, 2, stage)
    cur = battery.get_partial_value(i, 3, stage)
    temp = battery.get_partial_value(i, 6, stage)
    # print(f"the every cycle cap len:{len(cap)}")
    cycle_data = {
      "cycle": i,
      "capacity": cap,
      "voltage": vol,
      "current": cur,
      "temperature": temp,
    }
    data_.append(cycle_data)

  return data_


def cal_xj_battery_soh_data(battery: Battery):
  caps = battery.get_capacity()
  max_cap = max(caps)
  print("the max cap is :", max_cap)
  sohs_dict = {}
  i = 0
  for cap in caps:
    soh = cap / max_cap
    soh = float(f"{soh:.3f}")
    sohs_dict[i] = soh
    i += 1
  return sohs_dict


def get_xj_batch_data(batch: str):
  batch_battery_data = get_xj_data_file(
    read_xj_data_files(xj_data_files_path, batches_arr), batch
  )
  return batch_battery_data

"""
TJU datasets
"""
tj_data_files_path = r"D:\1New\Coding\Datasets\TJU"
tj_batches_arr = ["Dataset_1_NCA_battery", "Dataset_2_NCM_battery"]


def read_tj_data_files(path: str, batches: list[str]):
  files_count = {}
  for batch in batches:
    current_path = os.path.join(path, batch)
    if not os.path.isdir(current_path):
      print(f"error: {current_path} is not exist.")
      break

    filenames = os.listdir(current_path)
    files_count[batch] = [
      os.path.join(current_path, filename) for filename in filenames
    ]

  return files_count

def get_tj_data_file(files_path: {str, list[str]}, batch: str):
  paths = files_path[batch]
  pd_datas = {}
  for path in paths:
    if os.path.isfile(path):
      # idx = path.split("\\")[-1].split(".")[0].split("-")[-1]
      battery = TJBattery(path)
      idx = battery.battery_name
      pd_datas[idx] = battery
  print(f"the {batch} and have {len(paths)} battery")
  return pd_datas


def get_tj_battery_data(battery: TJBattery):
  # name = battery.battery_name
  cycle_idxs = battery._get_cycle_index()
  data_ = []
  for idx in cycle_idxs:
    # cap = battery.get_value(idx, )
    vol = battery.get_value(idx, "Ecell/V")
    cur = battery.get_value(idx, "<I>/mA")
    rel_vol = battery.get_rel_voltage(idx)
    ch_rel_vol = [v for v in rel_vol if v > 3.5]
    dch_rel_vol = [v for v in rel_vol if v < 3.5]
    cycle_data = {
      "cycle": idx,
      "voltage": vol,
      "current": cur,
      "rel_voltage":rel_vol,
      "charge_rel_voltage":ch_rel_vol,
      "discharge_rel_voltage":dch_rel_vol,
    }
    data_.append(cycle_data)

  return data_

def cal_tj_soh_battery(battery:TJBattery):
  caps = battery.get_degradation_trajectory()
  max_cap = max(caps)
  print("the max cap is :", max_cap)
  sohs_dict = {}
  i = 0
  for cap in caps:
    soh = cap / max_cap
    soh = float(f"{soh:.3f}")
    sohs_dict[i] = soh
    i += 1
  return sohs_dict
  
def resample(series, new_length):
  x_original = np.linspace(0,1, len(series))
  x_new = np.linspace(0, 1, new_length)
  f = interp1d(x_original, series, kind='linear')
  return f(x_new)

def get_tj_all_datas(batch_name:str):
  tj_all_files = read_tj_data_files(tj_data_files_path, tj_batches_arr)
  batch_all_datas = get_tj_data_file(tj_all_files, batch_name)
  return  batch_all_datas
  

"""
  测试代码
"""
#
# if __name__ == "__main__":
#   tj_files = read_tj_data_files(tj_data_files_path, tj_batches_arr)
#   battery_datas_1 = get_tj_data_file(tj_files, "Dataset_1_NCA_battery")
#
#   for _, battery in battery_datas_1.items():
#     battery_data = get_tj_battery_data(battery)
#     battery_soh = cal_tj_soh_battery(battery)
#     for cycle in battery_data:
#       vol = cycle["voltage"]
#       cur = cycle["current"]
#       print("the cycle data: ", vol, cur)
      
#   datas = get_xj_data_file(
#     read_xj_data_files(xj_data_files_path, batches_arr), "Batch-1"
#   )
#   #   # charge_datas = get_xj_battery_charge_data(batch1_all_battery_data['1'])
#   #   # print("this is the 1 battery ", charge_datas)
#   #   # print(batch1_all_battery_data['1'].get_one_cycle_description(1))
#   battery1 = datas[""]

#   bdata_ = get_xj_battery_data(battery1, 1)
#   fusion_datas = []
#   for cycle in bdata_:
#     i = cycle["cycle"]-1
#     # cap_ = bdata_[i]["capacity"]
#     vol_ = bdata_[i]["voltage"]
#     cur_ = bdata_[i]["current"]
#     temp_ = bdata_[i]["temperature"]
#     # print("the vol", vol)
#
#     # cap_ = gaf(cap_)
#     vol_, cur_, temp_ = gaf(vol_), gaf(cur_), gaf(temp_)
#     fusion_data = fusion(vol_, cur_, temp_)
#     print("the fusion data", fusion_data.shape)
#     plt.subplot(1,4,4)
#     plt.imshow(fusion_data)
#     plt.title("128x128x3 image")
#     plt.show()
#     plt.close()
#     fusion_datas.append(fusion_data)

  
  # print(fusion_datas)
#   # charge_datas = get_xj_battery_data(battery1, 1)
#   sohs = cal_xj_battery_soh_data(battery1)
#   print(sohs)
# print(charge_datas)
# print(battery1.battery_name)
# print(battery1.get_descriptions())
# battery1.get_degradation_trajectory()
# print(battery1.get_degradation_trajectory())
# print(battery1.get_capacity())

# get_xj_data_file(
#   read_xj_data_files(xj_data_files_path,batches_arr),
#   "Batch-1"
# )
#   Vols_data = read_dfs_by_cycle_toVol("XQ-11", dfs)
#   print("the Vols: ", Vols_data)
#   print("the dfs: ", dfs)

# Caps = read_dfs_by_cycle_toCap("XQ-11", dfs)
# # Times = read_dfs_by_cycle_toTime("XQ-11", dfs)
# # # print(Times)
# # norm_caps = []
# scaler = MinMaxScaler(feature_range=(-1,1))
# for cap in Caps:
#   norm_cap = scaler.fit_transform(cap.reshape(-1,1)).flatten()
#   gaf = GramianAngularField(method="summation", image_size=len(norm_cap))
#   gaf_img = gaf.fit_transform(norm_cap.reshape(-1,1))
#   print("the gaf_img is ",gaf_img)
