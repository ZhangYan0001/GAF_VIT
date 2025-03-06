import json

import data.get_feature as gf

df_keys = gf.df_keys
dfs = gf.dfs
def get_soh_labels(df_keys:[]):
  SOH_Labels = {}
  for df_key in df_keys:
    soh_label = gf.read_dfs_by_cycle_toSOH(df_key, dfs)
    SOH_Labels[df_key] = soh_label

  return SOH_Labels

# SOH_Labels = get_soh_labels(df_keys)
# with open("soh_data.json", "w", encoding="utf-8") as f:
#   json.dump(SOH_Labels, f, ensure_ascii=False, indent=4)
# print(SOH_Labels)
# print(len(SOH_Labels))


