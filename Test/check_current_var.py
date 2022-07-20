import numpy as np
import matplotlib.pyplot as plt
from utils.Entropy import Moving_Avg_Filter
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import operator
from scipy import integrate


## 5E mean 1.4, var 0.45
## 6F mean 2.2, var 1.0
## 7G mean 1.6, var 0.9


check_capacity1 = np.load('../data/battery_device/Data_Excel_1224/1_A_Battery/capacity.npy', allow_pickle=True)
check_capacity2 = np.load('../data/battery_device/Data_Excel_1224/2_B_Battery/capacity.npy', allow_pickle=True)
check_capacity3 = np.load('../data/battery_device/Data_Excel_1224/3_C_Battery/capacity.npy', allow_pickle=True)
check_capacity5 = np.load('../data/battery_device/Data_Excel_1224/5_E_Battery/capacity.npy', allow_pickle=True)
check_capacity6 = np.load('../data/battery_device/Data_Excel_1224/6_F_Battery/capacity.npy', allow_pickle=True)
check_capacity7 = np.load('../data/battery_device/Data_Excel_1224/7_G_Battery/capacity.npy', allow_pickle=True)

check_constant_current1 = np.load('../data/battery_device/Data_Excel_1224/1_A_Battery/discharge_current.npy', allow_pickle=True)
check_constant_current2 = np.load('../data/battery_device/Data_Excel_1224/2_B_Battery/discharge_current.npy', allow_pickle=True)
check_constant_current3 = np.load('../data/battery_device/Data_Excel_1224/3_C_Battery/discharge_current.npy', allow_pickle=True)
check_constant_current5 = np.load('../data/battery_device/Data_Excel_1224/5_E_Battery/discharge_current.npy', allow_pickle=True)
check_constant_current6 = np.load('../data/battery_device/Data_Excel_1224/6_F_Battery/discharge_current.npy', allow_pickle=True)
check_constant_current7 = np.load('../data/battery_device/Data_Excel_1224/7_G_Battery/discharge_current.npy', allow_pickle=True)

check_constant_entropy1 = np.load('../data/battery_device/Data_Excel_1224/1_A_Battery/discharge_Charge.npy', allow_pickle=True)     #discharge_Entropy
check_constant_entropy2 = np.load('../data/battery_device/Data_Excel_1224/2_B_Battery/discharge_Charge.npy', allow_pickle=True)
check_constant_entropy3 = np.load('../data/battery_device/Data_Excel_1224/3_C_Battery/discharge_Charge.npy', allow_pickle=True)
check_constant_entropy5 = np.load('../data/battery_device/Data_Excel_1224/5_E_Battery/discharge_Charge.npy', allow_pickle=True)
check_constant_entropy6 = np.load('../data/battery_device/Data_Excel_1224/6_F_Battery/discharge_Charge.npy', allow_pickle=True)
check_constant_entropy7 = np.load('../data/battery_device/Data_Excel_1224/7_G_Battery/discharge_Charge.npy', allow_pickle=True)

check_constant_time = np.load('../data/battery_device/Data_Excel_1224/8_H_Battery/discharge_time_all.npy', allow_pickle=True)
check_constant_voltage = np.load('../data/battery_device/Data_Excel_1224/1_A_Battery/discharge_data.npy', allow_pickle=True)

for i in range(len(check_constant_current3)-1, 0, -1):
    if len(check_constant_current3[i]) == 0:
        check_constant_current3 = np.delete(check_constant_current3, i)
        check_capacity3 = np.delete(check_capacity3, i)
    elif check_constant_current3[i][0] > -2 or check_constant_current3[i][0] < -2.4:
        check_constant_current3 = np.delete(check_constant_current3, i)
        check_capacity3 = np.delete(check_capacity3, i)


current_mean1 = []
current_var1 = []

current_mean2 = []
current_var2 = []

current_mean3 = []
current_var3 = []

current_mean5 = []
current_var5 = []

current_mean6 = []
current_var6 = []

current_mean7 = []
current_var7 = []

for i in range(len(check_constant_current1)):
    current_mean1.append(np.average(check_constant_current1[i]))
    current_var1.append(np.std(check_constant_current1[i]))

for i in range(len(check_constant_current2)):
    current_mean2.append(np.average(check_constant_current2[i]))
    current_var2.append(np.std(check_constant_current2[i]))

for i in range(len(check_constant_current3)):
    current_mean3.append(np.average(check_constant_current3[i]))
    current_var3.append(np.std(check_constant_current3[i]))

for i in range(len(check_constant_current5)):
    current_mean5.append(np.average(check_constant_current5[i]))
    current_var5.append(np.std(check_constant_current5[i]))

for i in range(len(check_constant_current6)):
    current_mean6.append(np.average(check_constant_current6[i]))
    current_var6.append(np.std(check_constant_current6[i]))

for i in range(len(check_constant_current7)):
    current_mean7.append(np.average(check_constant_current7[i]))
    current_var7.append(np.std(check_constant_current7[i]))


check_SOH1 = check_capacity1/3.25
check_SOH2 = check_capacity2/3.25
check_SOH3 = check_capacity3/3.25
check_SOH5 = check_capacity5/3.25
check_SOH6 = check_capacity6/3.25
check_SOH7 = check_capacity7/3.25

for i in range(len(check_SOH1)):
    check_SOH1[i] = round(check_SOH1[i], 3)
for i in range(len(check_SOH2)):
    check_SOH2[i] = round(check_SOH2[i], 3)
for i in range(len(check_SOH3)):
    check_SOH3[i] = round(check_SOH3[i], 3)
for i in range(len(check_SOH5)):
    check_SOH5[i] = round(check_SOH5[i], 3)
for i in range(len(check_SOH6)):
    check_SOH6[i] = round(check_SOH6[i], 3)
for i in range(len(check_SOH7)):
    check_SOH7[i] = round(check_SOH7[i], 3)


plt.figure()
plt.plot(check_constant_voltage[0])
plt.show()



SOH1_dic = {}
SOH2_dic = {}
SOH3_dic = {}
SOH5_dic = {}
SOH6_dic = {}
SOH7_dic = {}

for i in range(len(check_SOH1)):
    try:
        if SOH1_dic.get(check_SOH1[i]) != None:
            SOH1_dic[check_SOH1[i]] = (SOH1_dic.get(check_SOH1[i]) + check_constant_entropy1[i]) / 2
        else:
            SOH1_dic[check_SOH1[i]] = check_constant_entropy1[i]
    except:
        continue
# random_idx = []
# delete_idx = []
#
# for i in range(len(check_SOH2)-1, 0, -1):
#     if current_mean2[i] < -1.6 or current_mean2[i] > -1.57:
#         delete_idx.append(i)
#     else:
#         random_idx.append(i)
#
# delete_idx.append(0)
#
# check_SOH2 = np.delete(check_SOH2, delete_idx)
# check_constant_entropy2 = np.delete(check_constant_entropy2, delete_idx)
# current_mean2 = np.delete(current_mean2, delete_idx)


for i in range(len(check_SOH2)):
    try:
        if SOH2_dic.get(check_SOH2[i]) != None:
            SOH2_dic[check_SOH2[i]] = (SOH2_dic.get(check_SOH2[i]) + check_constant_entropy2[i]) / 2
        else:
            SOH2_dic[check_SOH2[i]] = check_constant_entropy2[i]
    except:
        continue

for i in range(len(check_SOH3)):
    try:
        if SOH3_dic.get(check_SOH3[i]) != None:
            SOH3_dic[check_SOH3[i]] = (SOH3_dic.get(check_SOH3[i]) + check_constant_entropy3[i]) / 2
        else:
            SOH3_dic[check_SOH3[i]] = check_constant_entropy3[i]
    except:
        continue

for i in range(len(check_SOH5)):
    try:
        if SOH5_dic.get(check_SOH5[i]) != None:
            SOH5_dic[check_SOH5[i]] = (SOH5_dic.get(check_SOH5[i]) + check_constant_entropy5[i]) / 2
        else:
            SOH5_dic[check_SOH5[i]] = check_constant_entropy5[i]
    except:
        continue

for i in range(len(check_SOH6)):
    try:
        if SOH6_dic.get(check_SOH6[i]) != None:
            SOH6_dic[check_SOH6[i]] = (SOH6_dic.get(check_SOH6[i]) + check_constant_entropy6[i]) / 2
        else:
            SOH6_dic[check_SOH6[i]] = check_constant_entropy6[i]
    except:
        continue

for i in range(len(check_SOH7)):
    try:
        if SOH7_dic.get(check_SOH7[i]) != None:
            SOH7_dic[check_SOH7[i]] = (SOH7_dic.get(check_SOH7[i]) + check_constant_entropy7[i]) / 2
        else:
            SOH7_dic[check_SOH7[i]] = check_constant_entropy7[i]
    except:
        continue

check_SOH1 = check_SOH1.tolist()
check_SOH1 = list(set(check_SOH1))
check_SOH1.sort(reverse=True)
check_SOH1.reverse()

check_SOH2 = check_SOH2.tolist()
check_SOH2 = list(set(check_SOH2))
check_SOH2.sort(reverse=True)
check_SOH2.reverse()

check_SOH3 = check_SOH3.tolist()
check_SOH3 = list(set(check_SOH3))
check_SOH3.sort(reverse=True)
check_SOH3.reverse()

check_SOH5 = check_SOH5.tolist()
check_SOH5 = list(set(check_SOH5))
check_SOH5.sort(reverse=True)
check_SOH5.reverse()

check_SOH6 = check_SOH6.tolist()
check_SOH6 = list(set(check_SOH6))
check_SOH6.sort(reverse=True)
check_SOH6.reverse()

check_SOH7 = check_SOH7.tolist()
check_SOH7 = list(set(check_SOH7))
check_SOH7.sort(reverse=True)
check_SOH7.reverse()

entropy1_2 = []
for i in range(len(check_SOH1)):
    entropy1_2.append(SOH1_dic[check_SOH1[i]])

entropy2_2 = []
for i in range(len(check_SOH2)):
    entropy2_2.append(SOH2_dic[check_SOH2[i]])

entropy3_2 = []
for i in range(len(check_SOH3)):
    try:
        entropy3_2.append(SOH3_dic[check_SOH3[i]])
    except:
        entropy3_2.append(SOH3_dic[check_SOH3[i-1]])
        print("error")

entropy5_2 = []
for i in range(len(check_SOH5)):
    entropy5_2.append(SOH5_dic[check_SOH5[i]])

entropy6_2 = []
for i in range(len(check_SOH6)):
    entropy6_2.append(SOH6_dic[check_SOH6[i]])

entropy7_2 = []
for i in range(len(check_SOH7)):
    entropy7_2.append(SOH7_dic[check_SOH7[i]])


# plt.figure()
#
# plt.plot(random_idx, current_mean2, label='channel 7')
# plt.plot(current_mean, label='channel 1')
#
# plt.ylabel('current average')
# plt.xlabel('time stamp')
# plt.legend()
# plt.ylim(-1.8, -1.40)
#
# plt.show()
#


plt.figure()
plt.plot(check_SOH1, entropy1_2, label='Battery 1')
plt.plot(check_SOH2[:-10], entropy2_2[:-10], label='Battery 2')
#plt.plot(check_SOH3, entropy3_2, label='channel 3')
plt.plot(check_SOH5[:-6], entropy5_2[:-6], label='Battery 3')
plt.plot(check_SOH6, entropy6_2, label='Battery 4')
plt.plot(check_SOH7, entropy7_2, label='Battery 5')
plt.ylabel('Charge Value')
plt.xlabel('SOH (%)')
plt.legend()
plt.gca().invert_xaxis()
plt.show()
