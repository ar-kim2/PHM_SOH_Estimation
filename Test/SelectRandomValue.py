import numpy as np
import random
import os


dir_path_CS2_change = "../data/battery_device/Data_Excel_1224"

#dir_path_CS2_change =  "../data/dis_current_constant/CS2_XX_0"
#dir_path_CS2_change =  "../data/Nasa_data/BatteryAgingARC_change"


battery = "1_A_Battery"
#battery = "CS2_38"
#battery = "B0028"

BatteryDataDir = os.path.join(dir_path_CS2_change, battery)

capacity = np.load(BatteryDataDir + '/capacity.npy', allow_pickle=True)
# discharge_data = np.load(BatteryDataDir + '/discharge_data.npy', allow_pickle=True)
# DC_list_current = np.load(BatteryDataDir + '/discharge_current.npy', allow_pickle=True)
# DC_time = np.load(BatteryDataDir + '/discharge_time_all.npy', allow_pickle=True)


data_len = len(capacity)
print("check : ", data_len)

data_len = int(np.round(data_len*0.6, 0))



sample_list = list(range(len(capacity)))
train_list = random.sample(sample_list[:-20], data_len)

test_list = list(set(sample_list) - set(train_list))
test_list.sort()
test_list = test_list[:-20]

np.save(BatteryDataDir + '/train_sample', train_list)
np.save(BatteryDataDir + '/test_sample', test_list)



# sample_list = list(range(len(capacity)))
# train_list = random.sample(sample_list[:-10], data_len)
#
# test_list = list(set(sample_list) - set(train_list))
# test_list.sort()
# test_list = test_list[:]
#
# np.save(BatteryDataDir + '/train_sample', train_list)
# np.save(BatteryDataDir + '/test_sample', test_list)