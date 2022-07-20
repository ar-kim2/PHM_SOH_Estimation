import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import operator

from scipy import integrate


charge_data = np.load('../data/battery_device/Data_Excel_ChargeData/Charge_data/charge_data.npy', allow_pickle=True)

print("shape : ", np.shape(charge_data))

time1 = []
time2 = []
time3 = []
for i in range(len(charge_data[1])):
    time1.append(i*30)

for i in range(len(charge_data[2])):
    time2.append(i*30)

for i in range(len(charge_data[3])):
    time3.append(i*30)

plt.figure()
plt.plot(time1, charge_data[1], label='Charge Current = 1A')
plt.plot(time2, charge_data[2],label= 'Charge Current = 2A')
#plt.plot(time3, charge_data[3],label= 'Charge Current = 3A')
plt.xlabel("Time stamp (s)")
plt.ylabel("Voltage (v)")
plt.legend()
plt.show()

#
# dir_path = "../data/dis_current_constant/K2_XX_0/K2_016"
#
# capacity = np.load(dir_path + '/capacity.npy', allow_pickle=True)
# discharge_data = np.load(dir_path + '/discharge_data.npy', allow_pickle=True)
# discharge_current_data = np.load(dir_path + '/discharge_current.npy', allow_pickle=True)
# charge_data = np.load(dir_path + '/charge_data.npy', allow_pickle=True)
# charge_current_data = np.load(dir_path + '/charge_current.npy', allow_pickle=True)
# last_cycle = np.load(dir_path + '/last_cycle.npy', allow_pickle=True)
#
# print("capacity : ", np.shape(capacity))
# print("discharge_data : ", np.shape(discharge_data))
# print("discharge_current_data : ", np.shape(discharge_current_data))
# print("charge_data : ", np.shape(charge_data))
# print("charge_current_data : ", np.shape(charge_current_data))
#
# print("last_cycle : ", last_cycle)
#
# # plt.figure()
# # plt.title("discharge voltage")
# # plt.plot(discharge_data[1])
# #
# # plt.figure()
# # plt.title("cpapcity")
# # plt.plot(capacity)
# # plt.show()
#
#
# CS2_34_entropy = np.load('../data/dis_current_constant/CS2_XX_0/CS2_34/concEntropy.npy', allow_pickle=True)
# CX2_34_entorpy = np.load('../data/dis_current_constant/CX2_XX_0/CX2_35/concEntropy.npy', allow_pickle=True)
# K2_16_entorpy = np.load('../data/dis_current_constant/K2_XX_0/K2_016/concEntropy.npy', allow_pickle=True)
#
# CS2_34_cap = np.load('../data/dis_current_constant/CS2_XX_0/CS2_34/capacity.npy', allow_pickle=True)
# CX2_34_cap = np.load('../data/dis_current_constant/CX2_XX_0/CX2_35/capacity.npy', allow_pickle=True)
# K2_16_cap = np.load('../data/dis_current_constant/K2_XX_0/K2_016/capacity.npy', allow_pickle=True)
#
# CS2_34_SOH = CS2_34_cap/1.1
# CX2_34_SOH = CX2_34_cap/1.35
# K2_16_SOH = K2_16_cap/2.6
#
# for i in range(len(CS2_34_SOH)):
#     CS2_34_SOH[i] = round(CS2_34_SOH[i],2)
# for i in range(len(CX2_34_SOH)):
#     CX2_34_SOH[i] = round(CX2_34_SOH[i],2)
# for i in range(len(K2_16_SOH)):
#     K2_16_SOH[i] = round(K2_16_SOH[i],2)
#
# CS2_dic = {}
# CX2_dic = {}
# K2_dic = {}
#
# for i in range(len(CS2_34_SOH)):
#     try:
#         CS2_dic[CS2_34_SOH[i]] = CS2_34_entropy[0][i][0]
#     except:
#         continue
#
# for i in range(len(CX2_34_SOH)):
#     try:
#         CX2_dic[CX2_34_SOH[i]] = CX2_34_entorpy[0][i][0]
#     except:
#         continue
#
# for i in range(len(K2_16_SOH)):
#     try:
#         K2_dic[K2_16_SOH[i]] = K2_16_entorpy[0][i][0]
#     except:
#         continue
#
#
# CS2_34_SOH = CS2_34_SOH.tolist()
# CS2_34_SOH = list(set(CS2_34_SOH))
# CS2_34_SOH.sort(reverse=True)
#
# CX2_34_SOH = CX2_34_SOH.tolist()
# CX2_34_SOH = list(set(CX2_34_SOH))
# CX2_34_SOH.sort(reverse=True)
#
# K2_16_SOH = K2_16_SOH.tolist()
# K2_16_SOH = list(set(K2_16_SOH))
# K2_16_SOH.sort(reverse=True)
#
# CS2_34_entropy2 = []
# for i in range(len(CS2_34_SOH)):
#     CS2_34_entropy2.append(CS2_dic[CS2_34_SOH[i]])
#
# CX2_34_entropy2 = []
# for i in range(len(CX2_34_SOH)):
#     CX2_34_entropy2.append(CX2_dic[CX2_34_SOH[i]])
#
# K2_16_entropy2 = []
# for i in range(len(K2_16_SOH)):
#     K2_16_entropy2.append(K2_dic[K2_16_SOH[i]])
#
# plt.figure()
# plt.plot(CS2_34_SOH, CS2_34_entropy2)
# plt.plot(CX2_34_SOH, CX2_34_entropy2)
# plt.plot(K2_16_SOH, K2_16_entropy2)
# plt.show()




# CS2_34_dic2 = sorted(CS2_dit.items(), key=operator.itemgetter(0))

# print(CS2_dit.keys())


# CS2_34_discharge_voltage = np.load('./data/test/dis_current_constant/CS2_XX_0/CS2_34/discharge_data.npy', allow_pickle=True)
# CX2_34_discharge_voltage = np.load('./data/test/dis_current_constant/CX2_XX_0/CX2_35/discharge_data.npy', allow_pickle=True)
#
# CS2_34_discharge_current = np.load('./data/test/dis_current_constant/CS2_XX_0/CS2_34/discharge_current.npy', allow_pickle=True)
# CX2_34_discharge_current = np.load('./data/test/dis_current_constant/CX2_XX_0/CX2_35/discharge_current.npy', allow_pickle=True)

# print(np.shape(CS2_34_entropy))

# plt.figure()
# plt.title("CS2_34, CX2_35 discharge voltage")
# plt.plot(CS2_34_discharge_voltage[0], label='CS2_34')
# plt.plot(CX2_34_discharge_voltage[0], label='CX2_35')
# plt.legend()
#
# plt.figure()
# plt.title("CS2_34, CX2_35 discharge current")
# plt.plot(CS2_34_discharge_current[0], label='CS2_34')
# plt.plot(CX2_34_discharge_current[0], label='CX2_35')
# plt.legend()
#
# plt.figure()
# plt.title("K2 entropy")
# plt.plot(K2_16_entorpy[0][:,0])
# plt.plot(K2_16_entorpy[0][:,1])
# plt.show()
# plt.figure()
# plt.title("CX2_34 entropy")
# plt.plot(CX2_34_entorpy[0][:,0])
# plt.plot(CX2_34_entorpy[0][:,1])
# plt.show()



# str = 'abc_def'
#
# if str.__contains__('abc'):
#     print("True")


#cycleInx = list(np.array(discharge_data)[:, 0])





# array = [[],["a", "b"], ["c", "d"]]
# arra2 = [1,2,3,4,5,6,6]
#
# a =  arra2.index(6, 6)



# array3 = [[1, 1, 1, 1],             #0
#           [1, 2, 1, 1],             #1
#           [1, 2, 1, 1],             #2
#           [1, 2, 1, 1],             #3
#           [1, 2, -1, 1],             #4
#           [1, 4, 1, 1],             #5
#           [1, 4, 1, 1],             #6
#           [1, 5, -1, 1],             #7
#           [1, 5, -1, 1],             #8
#           [1, 5, -1, 1],             #9
#           [1, 5, -1, 1],             #10
#           [1, 6, 1, 1],             #11
#           [1, 6, 1, 2],             #12
#           [2, 1, 1, 1],             #13
#           [2, 2, 1, 1],             #14
#           [2, 2, 1, 1],             #15
#           [2, 2, 1, 1],             #16
#           [2, 3, 1, 1],             #17
#           [2, 4, 1, 1],             #18
#           [2, 5, 1, 1],             #19
#           [2, 5, 1, 1],             #20
#           [2, 5, 1, 1],             #21
#           [2, 5, 1, 1],             #22
#           [2, 6, 1, 1]]             #23

#
#
# cycleList = list(np.array(array3)[:, 0])
# firstCycleIdx = cycleList.index(1)
# secondCycleIdx = cycleList.index(2)
#
# firstCycleData = list(np.array(array3)[firstCycleIdx:secondCycleIdx])
#
# stepList = list(np.array(firstCycleData)[:, 1])
#
# min = min(stepList)
# max = max(stepList)
#
# step_index = []
# for i in range(min, max):
#     try:
#         index = stepList.index(i)
#     except:
#         continue
#     step_index.append(index)
# step_index.append(len(stepList)-1)
#
# max1 = 0
# max2 = 0
#
# prev_len = 0
# for i in range(len(step_index)-1):
#     len = step_index[i+1]-step_index[i]
#     if len > prev_len:
#         max2 = max1
#         prev_len = len
#         max1 = i
#
#
# # print(firstCycleIdx)
# # print(secondCycleIdx)
# # print(firstCycleData)
# print(stepList)
# print(step_index)
# print(max1)
# print(max2)
#
#
# stepList = list(np.array(array3)[:, 0])

#
# print(len(array))
#
# for i in range(len(array)-1):
#     if len(array[i]) == 0:
#         array.pop(i)
#
# for i, value in enumerate(array):
#     print("i : ", i, " value : ", value)



# arr = [1,2,3,4,5]
# arr2 = [6,7,8,9,10,11,12,13,14,15]
# arr3 = np.append(arr, arr2)
# print(np.append(arr3, arr2))
# #arr = [[1],[2],[3],[4],[5]]
#
#
# print(np.round(1.335, 1))
#
#
# for i in range(1, 3):
#     print(i)

# tmp_arr = []
# for i in range(5):
#     tmp_arr.append([arr[i]])

# arr = np.array(arr)
# tmp_arr = arr[:, np.newaxis]

# tmp_arr = np.expand_dims(arr, axis=1)
# #tmp_arr = np.reshape(arr, 5, 1)
#
# print(np.shape(tmp_arr))
#
# print(tmp_arr)
# print(np.append(arr, arr2))



a= np.array([[1,2,3], [11,22,33], [111, 222, 333]])
b = [[4,5,6],[44,55,66]]
#b.extend(a)


c = [0.11, 0.22, 0.55, 1.1, 1.65, 2.2]

print("1 " ,np.mean(c))
print("2 ", np.std(c))


print(c[1:1+3])


# c = np.concatenate((a, b),axis=1 )
#
#
# DC_newaxis = np.array(a)[:, np.newaxis]
# print(DC_newaxis)
#
#
#
#
#
# x = np.array([1,2,3,4,5])
# y = [1,1,1,1,2]
#
# np.average(x)
#
#
#
# plt.figure()
# plt.plot(x, y)
# plt.ylim(0,2.5)
# plt.show()
#
# integral = integrate.cumtrapz(y,x, 0.5)
# #print(integral)
#
#
# x = [1.89, 1.82, 1.92]
# x = np.multiply(x, 10)
# x= np.trunc(x)
# x = x/10
#
#
# x = np.linspace(0,10,3)
#
#
# test = [1, 2 , 3]
# print("1 ", test)
# print("2 : ", test.append(4))
# print("3 : ", test)
# test = []
# print("4 : ", test)


for i in range(5):
    print(i)


import tensorflow as tf

# nanTest = [np.Nan]
#
# if nanTest[0] == np.Nan:
#     print("Nan")

a = [[1,2],[2,1]]
b=[[4,1],[2,2]]
print("Cross check :", np.cross(a,b))