import json
import pandas as pd

datas = []

with open("./all_xj_batch_1_results.jsonl", "r", encoding="utf-8") as f:
  for line in f:
    datas.append(json.loads(line))
    

for data in datas:
  
  df = pd.DataFrame(data)

  print(df.head())