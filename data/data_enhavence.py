import numpy as np
from scipy.interpolate import interp1d
import data.get_feature as gf

class TimeSeriesAugmentor:
  def __init__(self, target_length=224, noise_std=0.01):
    self.target_length = target_length
    self.noise_std = noise_std
  
  def add_noise(self, ts):
    ts = np.array(ts)
    std = np.std(ts)
    noise = np.random.normal(0, self.noise_std * std, size = ts.shape)
    return noise + ts
  
  def time_scale(self, ts, scale_factor=None):
    if scale_factor is None:
      scale_factor = np.random.uniform(0.8, 1.2)
      
    old_len = len(ts)
    new_len = int(old_len * scale_factor)
    
    x_old = np.linspace(0, 1, old_len)
    x_new = np.linspace(0, 1, new_len)
    f = interp1d(x_old, ts, kind="linear")
    ts_scaled = f(x_new)
    
    return ts_scaled
  
  def resize(self, ts):
    curr_len = len(ts)
    x_old = np.linspace(0, 1, curr_len)
    x_new = np.linspace(0, 1, self.target_length)
    f = interp1d(x_old, ts, kind="linear")
    return f(x_new)
  
  def augment_pair(self, ts):
    ts1 = self.add_noise(ts)
    ts1 = self.resize(ts1)
    
    ts2 = self.time_scale(ts)
    ts2 = self.resize(ts2)
    
    return ts1, ts2
    
"""
  testing
"""
if __name__ == "__main__":
  
  battery_datas1 = gf.get_tj_all_datas("Dataset_1_NCA_battery")
  for k, battery in battery_datas1.items():
    print(k)
    print(battery)
    battery_data = gf.get_tj_battery_data(battery)
    data_augmentor = TimeSeriesAugmentor()
    for cycle_data in battery_data:
      ch_rel_voltage = cycle_data["charge_rel_voltage"]
      ch1, ch2 = data_augmentor.augment_pair(ch_rel_voltage)
      print(f"the ch1:{ch1}, and the len: {len(ch1)}")
      print(f"the ch2:{ch2}, and the len: {len(ch2)}")
      break
    break