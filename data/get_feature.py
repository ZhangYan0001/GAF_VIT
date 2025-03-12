import os.path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from data.xjbattery import Battery

# dfs = {}

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
    sohs = np.array(list(df_lim["SoH"])).reshape(-1)
    sohs_avg = round(np.mean(sohs), 6)
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


def get_soh_labels(keys: []):
  SOH_Labels = {}
  for df_key in keys:
    soh_label = read_dfs_by_cycle_toSOH(df_key, dfs)
    SOH_Labels[df_key] = soh_label

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
      idx = path.split("\\")[-1].split(".")[0].split("-")[-1]
      mat_datas[idx] = Battery(path)
  print(f"the {batch} and have {len(paths)} battery")
  return mat_datas


def get_xj_battery_vol_data(battery: Battery):
  name = battery.battery_name
  cycle_life = battery.cycle_life
  voltages = []
  for i in range(cycle_life):
    voltage = battery.get_partial_value(
      i,
      2,
      1,
    )
    print(f"the every cycle voltage len:{len(voltage)}")
    voltages.append(voltage)

  print(f"{name} voltage data and the len is {len(voltages)}")
  return voltages


def get_xj_battery_charge_data(battery: Battery):
  name = battery.battery_name
  cycle_life = battery.cycle_life
  charge_data = []
  for i in range(cycle_life):
    cap = battery.get_capacity()[i]
    vol = battery.get_partial_value(i, 2, 1)
    cur = battery.get_partial_value(i, 3, 1)
    temp = battery.get_partial_value(i, 6, 1)
    # print(f"the every cycle cap len:{len(cap)}")
    cycle_data = {
      "cycle": i+1,
      "capacity":cap,
      "voltage":vol,
      "current":cur,
      "temperature":temp
    }
    charge_data.append(cycle_data)

  return charge_data

# def cal_xj_battery_charge_soh_data(battery:Battery):
#   init_cap = battery.get_partial_value(1,4,1)
#   init_cap = max(init_cap)
#   sohs = []
#   for i in range(1, battery.cycle_life):
  
  


batch1_all_battery_data = get_xj_data_file(
  read_xj_data_files(xj_data_files_path, batches_arr), "Batch-1"
)

"""
  测试代码
"""

if __name__ == "__main__":
  # print(
  #   "this is files path ",
  #   read_xj_data_files(xj_data_files_path, batches_arr)
  #
  # datas = get_xj_data_file(
  #   read_xj_data_files(xj_data_files_path, batches_arr), "Batch-1"
  # )
  # charge_datas = get_xj_battery_charge_data(batch1_all_battery_data['1'])
  # print("this is the 1 battery ", charge_datas)
  # print(batch1_all_battery_data['1'].get_one_cycle_description(1))
  battery1 = batch1_all_battery_data['1']
  print(battery1.get_descriptions())
  # battery1.get_degradation_trajectory()
  # print(battery1.get_degradation_trajectory())
  print(battery1.get_capacity())
  
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
