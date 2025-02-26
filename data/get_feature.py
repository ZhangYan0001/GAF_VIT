import numpy as np

import data.get_data as gd

df_keys = [
  "XQ-11",
  "XQ-12",
  "XQ-14",
  "XQ-15",
  "XQ-16",
  "XQ-17",
  "XQ-18",
]
dfs = gd.get_df_data(gd.data_files_path)


def read_dfs_by_cycle_toCap(df_key, dfs: dict):
  df = dfs[df_key]
  cycles = list(set(df["循环"]))
  Caps = []
  for c in cycles:
    if c in [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]:
      continue
    df_lim = df[df["循环"] == c]
    Cap = np.array(list(df_lim["容量(Ah)"])).reshape(-1)
    Caps.append(Cap)

  return np.array(Caps, dtype=object)


def read_dfs_by_cycle_toTime(df_key, dfs: dict):
  df = dfs[df_key]
  cycles = list(set(df["循环"]))
  times = []
  for c in cycles:
    if c in [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]:
      continue
    df_lim = df[df["循环"] == c]
    time = np.array(list(df_lim["time"])).reshape(-1)
    times.append(time)
  return np.array(times, dtype=object)


def normalize_data(df, col: str):
  min_d = df[str].min()
  max_d = df[str].max()
  return (df[str] - min_d) / (max_d - min_d)


def read_dfs_by_cycle_toSOH(df_key, dfs: dict):
  df = dfs[df_key]
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
