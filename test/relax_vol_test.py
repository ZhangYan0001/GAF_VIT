"""
画图 文件
"""

from data.xjbattery import Battery
from data.get_feature import get_xj_battery_data, get_xj_batch_data, cal_xj_battery_soh_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.cm as cm
from scipy.stats import spearmanr

batch1_data = get_xj_batch_data("Batch-1")



def draw_rel_voltage():
  for bty_i, bty_d in batch1_data.items():
    cycle = bty_d.cycle_life
    
    vol_charge_l = []
    for i in range(2, cycle):
      rel_voltage_charge = bty_d.get_original_partial_value(i, 2, 2)
      # mean_rel_vol = np.mean(rel_voltage_charge)
      mean_rel_vol = np.median(rel_voltage_charge)
      vol_charge_l.append(mean_rel_vol)
    
    smooth_vol_charge_l = savgol_filter(vol_charge_l, window_length=11, polyorder=3)
    x_ = np.array(smooth_vol_charge_l)
    y_ = np.array(list(cal_xj_battery_soh_data(bty_d).values())[1:-1])

    r_ = np.corrcoef(x_, y_)[0,1]
    s_r_ , s_p_ = spearmanr(x_.tolist(), y_.tolist())
    print(f"Pearson 相关性系数 r = {r_:.3f}")
    print(f"Spearmanr 相关性系数 r = {s_r_:.3f}, P value = {s_p_:.3f}")
    plt.plot(smooth_vol_charge_l, label=f"Battery {bty_i}")
    # plt.ylim(4.15,4.20)
  
  plt.title("Relaxation Voltage Curves for Different Batteries")
  plt.xlabel("Cycle Index")
  plt.ylabel("median Relaxation Voltage(V)")
  plt.legend()
  plt.grid(True)
  plt.show()
    
def draw_rel_voltage_by_cycle():
  for _, bty_d in batch1_data.items():
    cycle = bty_d.cycle_life
    
    colormap = cm.viridis
    norm = plt.Normalize(vmin=2, vmax=cycle+1)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    
    fig, ax1 = plt.subplots(figsize=(10,5))
    
    
    for i in range(2, cycle):
      vol = bty_d.get_partial_value(i, 2, 2)
      times = list(range(len(vol)))
      vol = savgol_filter(vol, window_length=11, polyorder=4)
      color = colormap(i / cycle)
      ax1.plot(times, vol, color=color, linewidth = 3, alpha=0.6)
      
    ax1.set_xlabel("Measurement Point")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Battery rel Voltage")
    
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.set_label("Cycle Number")
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def draw_voltage_current():
  for _ , bty_d in batch1_data.items():
    cycle =  bty_d.cycle_life
    vols = []
    curs = []
    
    for i in range(2, cycle+1):
      vol = bty_d.get_value(i, 2)
      cur = bty_d.get_value(i, 3)
      # vols.extend(vol)
      # curs.extend(cur)
      
      vol = savgol_filter(vol, window_length=51, polyorder=4)
      cur = savgol_filter(cur, window_length=51, polyorder=4)
      
      rel_vol_charge = bty_d.get_original_partial_value(i, 2, 2)
      rel_vol_discharge = bty_d.get_original_partial_value(i, 2, 4)

    
      times_vols = list(range(len(vol)))
      times_curs = list(range(len(cur)))
      times_rel_vols_charge = list(range(len(rel_vol_charge)))
      times_rel_vols_discharge = list(range(len(rel_vol_discharge)))

      # 绘制曲线
      fig, ax1 = plt.subplots(figsize=(10,5))
      ax1.set_xlabel("times")
      ax1.set_ylabel("Voltage(V)", color='b')
      ax1.plot(times_vols, vol, label="Voltage (V)", color = 'b', marker='o', linestyle='-', linewidth=1, markersize=1)
      
      ax2 =ax1.twinx()
      ax2.set_ylabel("Current (A)", color="r")
      ax2.plot(times_curs, cur, label="Current (A)", color='r', marker='s', linestyle='--', linewidth=1, markersize=1)
      ax2.tick_params(axis='y', labelcolor='r')
      
      ax_inset = ax1.inset_axes([0.5, 0.2, 0.4, 0.3])
      ax_inset.plot(times_rel_vols_charge, rel_vol_charge, 'b-', label="rel voltage", linewidth=1)
      ax_inset.set_xlabel("times")
      ax_inset.set_ylabel("Voltage (V)")
      ax_inset.grid(True, linestyle=":", alpha=0.5)
      ax_inset.legend(fontsize=8)

      
      plt.title(f'Voltage and Current over {cycle} Cycles')
      fig.tight_layout()
      plt.grid(True, linestyle='--', alpha=0.6)
      plt.show()
      

def plt_img_rel():
  for battery_idx, battery_data in batch1_data.items():
    cycle = battery_data.cycle_life
    for i in range(1, cycle):
      rel_voltage_charge = battery_data.get_original_partial_value(i, 2, 2)
      rel_voltage_discharge = battery_data.get_original_partial_value(i, 2, 4)
      rel_current_charge = battery_data.get_original_partial_value(i, 3, 2)
      rel_current_discharge = battery_data.get_original_partial_value(i, 3, 4)
      time1 = battery_data.get_original_partial_value(i, 1, 2)
      time2 = battery_data.get_original_partial_value(i, 1, 4)
      x_l = np.linspace(0, len(rel_voltage_charge), len(rel_voltage_charge))
      x_l2 = np.linspace(0, len(rel_voltage_discharge), len(rel_voltage_discharge))
      x_l3 = np.linspace(0, len(rel_current_charge), len(rel_current_charge))
      x_l4 = np.linspace(0, len(rel_current_discharge), len(rel_current_discharge))
      plt.subplot(2, 2, 1)
      plt.xlabel("time")
      plt.ylabel("voltage(V)")
      plt.plot(x_l, rel_voltage_charge, color="blue", linestyle="-")
      plt.subplot(2, 2, 2)
      plt.plot(x_l2, rel_voltage_discharge, color="red", linestyle="-")
      plt.subplot(2, 2, 3)
      plt.plot(x_l3, rel_current_charge, color="green", linestyle="-")
      plt.subplot(2, 2, 4)
      plt.plot(x_l4, rel_current_discharge, color="yellow", linestyle="-")
      plt.xlabel("time")
      plt.ylabel("current(A)")
      
      plt.legend()
      plt.show()

if __name__ == "__main__":
  draw_rel_voltage()
  # draw_rel_voltage_by_cycle()
  # draw_voltage_current()