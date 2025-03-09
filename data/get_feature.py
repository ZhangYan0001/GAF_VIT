import numpy as np
import pandas as pd

# dfs = {}


def read_file(file_path: str):
  df = []
  if file_path.split('.')[-1] == 'xlsx':
    df = pd.read_excel(file_path)
  else:
    df = pd.read_csv(file_path)
  return df


def get_df_data(file_path: str, file_names: []):
  dfs = {}
  for i in file_names:
    dfs[i.split('.')[0][:5]] = read_file(file_path + '\\' + i)
  return dfs


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
  "XQ-18-25-1C-pre.xlsx"
]

data_files_path = r"F:\New\Coding\Datasets\data"
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


if __name__ == "__main__":
  Vols_data = read_dfs_by_cycle_toVol("XQ-11", dfs)
  print("the Vols: ", Vols_data)
  print("the dfs: ", dfs)

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
