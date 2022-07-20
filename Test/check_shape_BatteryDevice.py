import numpy as np
import matplotlib.pyplot as plt
from utils.Entropy import Moving_Avg_Filter
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import operator
from scipy import integrate

check_constant_capacity = np.load('../data/battery_device/Data_Excel_210320/1_A_Battery/capacity.npy', allow_pickle=True)
check_constant_capacity2 = np.load('../data/battery_device/Data_Excel_210320/7_G_Battery/capacity.npy', allow_pickle=True)
check_constant_capacity3 = np.load('../data/battery_device/Data_Excel_1224/6_F_Battery/capacity.npy', allow_pickle=True)
check_constant_capacity4 = np.load('../data/battery_device/Data_Excel_1224/7_G_Battery/capacity.npy', allow_pickle=True)
check_constant_voltage = np.load('../data/battery_device/Data_Excel_1224/6_F_Battery/charge_data.npy', allow_pickle=True)
check_constant_current = np.load('../data/battery_device/Data_Excel_0910/5_E_Battery/discharge_current.npy', allow_pickle=True)
check_constant_time = np.load('../data/battery_device/Data_Excel_0910/8_H_Battery/discharge_time_all.npy', allow_pickle=True)

print("CHECK 3.2", np.where(check_constant_capacity<2.9))
print("CHECK 3.2", np.where(check_constant_capacity2<2.9))
concat = [] #check_constant_capacity2[:50]
concat2 = []
for i in range(64):
    concat.append(check_constant_capacity2[i])

for i in range(len(check_constant_capacity)):
    concat.append(check_constant_capacity[i])

for i in range(74):
    concat2.append(check_constant_capacity2[i])

for i in range(240, 1051):
    concat2.append(check_constant_capacity2[i])


plt.figure()
plt.plot(check_constant_capacity[:], label='Battery1 = m=1.6A, s=0')
#plt.plot(check_constant_capacity2[62:],  label='Battery2 = m=1.6A, s=0.92')
plt.plot(concat2,  label='Battery2 = m=1.6A, s=0.92')
plt.legend()
plt.show()

# plt.figure()
# plt.plot(check_constant_capacity2, label='capacity5')
#
# plt.legend()
# plt.title("capacity 1106 5")
#
# plt.figure()
# plt.plot(check_constant_capacity3, label='capacity')
# plt.legend()
# plt.title("capacity 1106 6")
#
# plt.figure()
# plt.plot(check_constant_capacity4, label='capacity')
# plt.legend()
# plt.title("capacity 1106 7")

plt.figure()
plt.plot(check_constant_voltage[1], label='voltage')
plt.legend()
plt.title("voltage  ")
#
# plt.figure()
# plt.plot(check_constant_current[3], label='current')
# plt.ylim(-4.5 , 0.5)
# plt.legend()
# plt.title("current  ")
plt.show()

