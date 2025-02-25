import pandas as pd

data_files_path = r"F:\New\Coding\Datasets\data"
files_name = [
  "XQ-11-25-1C-pre.xlsx",
  "XQ-12-25-1C-pre.xlsx",
  "XQ-14-25-1C-pre.xlsx",
  "XQ-15-25-1C-pre.xlsx",
  "XQ-16-25-1C-pre.xlsx",
  "XQ-17-25-1C-pre.xlsx",
  "XQ-18-25-1C-pre.xlsx"
]
def read_file(file_path: str):
  df = []
  if file_path.split('.')[-1] == 'xlsx':
    df = pd.read_excel(file_path)
  else:
    df = pd.read_csv(file_path)
  return df

def get_df_data(file_path: str):
  dfs = {}
  for i in files_name:
    dfs[i.split('.')[0][:5]] = read_file(file_path+'\\'+i)
  return dfs
# test
# dfs = get_df_data()
# print(dfs)
