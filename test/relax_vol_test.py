from data.xjbattery import Battery
from data.get_feature import get_xj_battery_data, get_xj_batch_data
import numpy as np
import matplotlib.pyplot as plt


batch1_data = get_xj_batch_data("Batch-1")


for battery_idx, battery_data in batch1_data.items():
  cycle = battery_data.cycle_life
  for i in range(1, cycle):
   rel_voltage_charge = battery_data.get_original_partial_value(i,2,2)
   rel_voltage_discharge = battery_data.get_original_partial_value(i,2,4)
   rel_current_charge = battery_data.get_original_partial_value(i,3,2)
   rel_current_discharge = battery_data.get_original_partial_value(i,3,4)
   time1 = battery_data.get_original_partial_value(i, 1, 2)
   time2 = battery_data.get_original_partial_value(i, 1, 4)
   x_l = np.linspace(0, len(rel_voltage_charge), len(rel_voltage_charge))
   x_l2 = np.linspace(0, len(rel_voltage_discharge), len(rel_voltage_discharge))
   x_l3 = np.linspace(0, len(rel_current_charge), len(rel_current_charge))
   x_l4 = np.linspace(0, len(rel_current_discharge), len(rel_current_discharge))
   plt.subplot(2,2,1)
   plt.xlabel("time")
   plt.ylabel("voltage(V)")
   plt.plot(x_l,rel_voltage_charge, color='blue', linestyle="-")
   plt.subplot(2,2,2)
   plt.plot(x_l2,rel_voltage_discharge, color='red', linestyle="-")
   plt.subplot(2,2,3)
   plt.plot(x_l3,rel_current_charge, color='green', linestyle="-")
   plt.subplot(2,2,4)
   plt.plot(x_l4,rel_current_discharge, color='yellow', linestyle="-")
   plt.xlabel("time")
   plt.ylabel("current(A)")

   plt.legend()
   plt.show()