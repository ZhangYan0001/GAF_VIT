"""
  三通道融合，图像转换
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField


def gaf(data):
  scaler = MinMaxScaler(feature_range=(0,1))
  data_normalized = scaler.fit_transform(data.reshape(-1,1)).flatten()
  print("data normalized: ", data_normalized)
  image_size = data.size
  
  gaf_data = GramianAngularField(
    image_size = image_size,
    method="summation",
    sample_range=(0,1)
  )
  gaf_data = gaf_data.fit_transform(data_normalized.reshape(1,-1))
  print("the gaf_data:", gaf_data)
  # print("the gaf_data shape", gaf_data.size)
  
  return gaf_data
  
  
def fusion(data1, data2, data3):
  fusion_data = np.stack([data1,data2,data3], axis=-1)
  print("the fusion_data: ", fusion_data)
  print("the fusion_data shape: ", fusion_data.shape)
  return fusion_data


# def three_channels_fusion(data1_, data2_, data3_):
#
#   fusion_tensor = []
#   for d1, d2, d3 in data1_, data2_, data3_:
#     d1_gaf = gaf(d1)
#     d2_gaf = gaf(d2)
#     d3_gaf = gaf(d3)
#
#     fusion_data = fusion(d1_gaf, d2_gaf, d3_gaf)
#     fusion_tensor.append(fusion_data)
#
#   fusion_tensor = np.array(fusion_tensor)
#   print("the fusion_tensor", fusion_tensor)
#   return fusion_tensor
  
"""
the test code
"""

# if __name__ == "__main__":
