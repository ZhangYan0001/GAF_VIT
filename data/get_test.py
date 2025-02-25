import data.get_feature as gf

SOHs = gf.read_dfs_by_cycle_toSOH(gf.df_keys[0],gf.dfs)

print("this is :",SOHs)
print(len(SOHs))
