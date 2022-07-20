import numpy as np
import os
import pandas as pd
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt



def ReadData(FileNameList, BatteryDataDir):
    D_Voltage = []
    C_Voltage = []
    D_Current = []
    C_Current = []
    Capacity = []
    SOH = []
    last_cycle = 0

    for FileName in FileNameList:
        if not FileName.endswith("xls") or FileName.endswith("xlsx"):
            continue

        print("Data Loading : {}".format(FileName))

        xlsxFile = os.path.join(BatteryDataDir, FileName)
        xls = pd.ExcelFile(xlsxFile)
        _data = pd.DataFrame()

        sheet_index = xls.sheet_names[1:]

        # for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
        _data = pd.concat([_data, df_xls], axis=0)

        last_index = _data['循环'][len(_data['循环']) - 1]

        for cycle in range(1, last_index+1):
            try:
                list(_data['循环'].values).index(cycle)

                cycle_data = _data.loc[_data['循环'] == cycle]

                discharge_data = cycle_data.loc[_data['状态'] == '恒流放电']
                charge_data = cycle_data.loc[_data['状态'] == '恒流恒压充电']

                D_Voltage.append(discharge_data['电压(V)'].values)
                D_Current.append(discharge_data['电流(mA)'].values)

                C_Voltage.append(charge_data['电压(V)'].values)
                C_Current.append(charge_data['电流(mA)'].values)

                whole_cap = cycle_data.loc[_data['状态'] == 'SOH']['容量(mAh)'].values[0]

                Capacity.append(whole_cap)
                SOH.append(whole_cap / 3250)

                last_cycle = last_cycle + 1
            except:
                continue

    return D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle


def ReadData_constant(FileNameList, BatteryDataDir):
    D_Voltage = []
    C_Voltage = []
    D_Current = []
    C_Current = []
    D_Time = []
    Capacity = []
    SOH = []
    last_cycle = 0

    for FileName in FileNameList:
        if not FileName.endswith("xls") or FileName.endswith("xlsx"):
            continue

        print("Data Loading : {}".format(FileName))

        xlsxFile = os.path.join(BatteryDataDir, FileName)
        xls = pd.ExcelFile(xlsxFile)
        _data = pd.DataFrame()

        sheet_index = xls.sheet_names[1:]

        # for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
        _data = pd.concat([_data, df_xls], axis=0)

        last_index = _data['循环'][len(_data['循环']) - 1]



        for cycle in range(1, last_index+1):
            try:
                list(_data['循环'].values).index(cycle)
                print("Check dsta : ", list(_data['循环'].values))
            except:
                print("Exception!! ")
                continue

            cycle_data = _data.loc[_data['循环'] == cycle]

            discharge_data = cycle_data.loc[_data['状态'] == '恒流放电']
            charge_data = cycle_data.loc[_data['状态'] == '恒流恒压充电']

            D_index_list = discharge_data['跳转'].values


            D_index = np.where(D_index_list == D_index_list[-1])[0][0]


            tmp_time = discharge_data['相对时间(h:min:s.ms)'].values[:D_index]

            for ti in range(len(tmp_time)):
                sec = int(tmp_time[ti][5:7])
                min = int(tmp_time[ti][2:4])
                hour = int(tmp_time[ti][0:1])

                tmp_time[ti] = sec + (min * 60) + (hour * 3600)

            cap1 = discharge_data['容量(mAh)'].values[D_index-1]
            cap2 = discharge_data['容量(mAh)'].values[-1]
            whole_cap = (cap1 + cap2) / 1000


            if len(Capacity) > 1 and whole_cap < (Capacity[-1] * 0.95):
                continue

            if cycle > 400 and whole_cap < (Capacity[-1] * 0.98):
                continue

            if cycle > 400 and whole_cap > (Capacity[-1] * 1.02):
                continue

            Capacity.append(whole_cap)
            SOH.append(whole_cap / 3.25)

            D_Voltage.append(discharge_data['电压(V)'].values[:D_index])
            D_Current.append(discharge_data['电流(mA)'].values[:D_index]/1000)

            D_Time.append(tmp_time)

            C_Voltage.append(charge_data['电压(V)'].values)
            C_Current.append(charge_data['电流(mA)'].values/1000)

            last_cycle = last_cycle + 1

    return D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle, D_Time

def ReadData_random(FileNameList, BatteryDataDir):
    D_Voltage = []
    C_Voltage = []
    D_Current = []
    C_Current = []
    D_Time = []
    C_Time = []
    Capacity = []
    SOH = []
    last_cycle = 0

    for FileName in FileNameList:
        if not FileName.endswith("xls") or FileName.endswith("xlsx"):
            continue

        print("Data Loading : {}".format(FileName))

        xlsxFile = os.path.join(BatteryDataDir, FileName)
        xls = pd.ExcelFile(xlsxFile)
        _data = pd.DataFrame()

        sheet_index = xls.sheet_names[1:]

        # for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
        _data = pd.concat([_data, df_xls], axis=0)

        last_index = _data['循环'][len(_data['循环']) - 1]

        for cycle in range(1, last_index+1):
            try:
                list(_data['循环'].values).index(cycle)
            except:
                continue

            cycle_data = _data.loc[_data['循环'] == cycle]

            discharge_data = cycle_data.loc[_data['状态'] == '模拟工步']
            discharge_data2 = cycle_data.loc[_data['状态'] == '恒流放电']
            charge_data = cycle_data.loc[_data['状态'] == '恒压充电']

            D_Voltage.append(discharge_data['电压(V)'].values)
            D_Current.append(discharge_data['电流(mA)'].values/1000)

            tmp_time = discharge_data['相对时间(h:min:s.ms)'].values

            for ti in range(len(tmp_time)):
                sec = int(tmp_time[ti][5:7])
                min = int(tmp_time[ti][2:4])
                hour = int(tmp_time[ti][0:1])

                tmp_time[ti] = sec + (min * 60) + (hour * 3600)

            D_Time.append(tmp_time)

            C_Voltage.append(charge_data['电压(V)'].values)
            C_Current.append(charge_data['电流(mA)'].values/1000)

            tmp_time = charge_data['相对时间(h:min:s.ms)'].values

            for ti in range(len(tmp_time)):
                sec = int(tmp_time[ti][5:7])
                min = int(tmp_time[ti][2:4])
                hour = int(tmp_time[ti][0:1])

                tmp_time[ti] = sec + (min * 60) + (hour * 3600)

            C_Time.append(tmp_time)

            cap1 = discharge_data['容量(mAh)'].values[-1]
            cap2 = discharge_data2['容量(mAh)'].values[-1]

            whole_cap = (cap1 + cap2)/1000

            Capacity.append(whole_cap)
            SOH.append(whole_cap / 3250)

            last_cycle = last_cycle + 1

    return D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle, D_Time, C_Time


def ReadData_capapcity(FileNameList, BatteryDataDir):
    D_Voltage = []
    C_Voltage = []
    D_Current = []
    C_Current = []
    D_Time = []
    C_Time = []
    Capacity = []
    SOH = []
    last_cycle = 0

    for FileName in FileNameList:
        if not FileName.endswith("xls") or FileName.endswith("xlsx"):
            continue

        print("Data Loading : {}".format(FileName))

        xlsxFile = os.path.join(BatteryDataDir, FileName)
        xls = pd.ExcelFile(xlsxFile)
        _data = pd.DataFrame()

        sheet_index = xls.sheet_names[1:]

        # for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[0])
        _data = pd.concat([_data, df_xls], axis=0)

        last_index = _data['循环序号'][len(_data['循环序号']) - 1]

        for cycle in range(1, last_index+1):
            try:
                list(_data['循环序号'].values).index(cycle)
            except:
                continue

            cycle_data = _data.loc[_data['循环序号'] == cycle]

            cap = cycle_data['放电容量(mAh)'].values[-1]

            whole_cap = (cap)/1000

            Capacity.append(whole_cap)
            SOH.append(whole_cap / 3.25)

    return Capacity, SOH



def ReadAbnormalData(FileName):
    D_Voltage = []
    D_Current = []

    if not FileName.endswith("xls") or FileName.endswith("xlsx"):
        return D_Voltage

    print("Data Loading : {}".format(FileName))

    xlsxFile = FileName
    xls = pd.ExcelFile(FileName)
    _data = pd.DataFrame()

    sheet_index = xls.sheet_names[1:]

    # for sheets in sheet_index:
    df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
    _data = pd.concat([_data, df_xls], axis=0)

    last_index = _data['循环'][len(_data['循环']) - 1]

    for cycle in range(1, last_index+1):
        try:
            list(_data['循环'].values).index(cycle)

            cycle_data = _data.loc[_data['循环'] == cycle]

            discharge_data = cycle_data.loc[_data['状态'] == '恒流放电']
            charge_data = cycle_data.loc[_data['状态'] == '恒流恒压充电']

            D_Voltage.append(discharge_data['电压(V)'].values)
            D_Current.append(discharge_data['电流(mA)'].values)

            C_Voltage.append(charge_data['电压(V)'].values)
            C_Current.append(charge_data['电流(mA)'].values)

            whole_cap = cycle_data.loc[_data['状态'] == 'SOH']['容量(mAh)'].values[0]

            Capacity.append(whole_cap)
            SOH.append(whole_cap / 3250)

            last_cycle = last_cycle + 1
        except:
            continue

    elapsed_time = _data['相对时间(h:min:s.ms)'].values

    first_time = elapsed_time[0].split(':')   #first_time[2]
    first_time = first_time[2].split('.')
    first_time = int(first_time[0])
    second_time = elapsed_time[1].split(':')
    second_time = second_time[2].split('.')
    second_time = int(second_time[0])

    if first_time < second_time:
        time_step = second_time-first_time
    else:
        time_step = first_time- second_time

    return D_Voltage, D_Current, time_step

def Density(list):
    all_prob = []
    len_list = []

    min_vol = np.min(list)
    max_vol = np.max(list)

    for idx in range(len(list)):
        hist, bin_edge = np.histogram(list[idx], bins=np.linspace(min_vol, max_vol, 17))     # np.linspace(2,5,17) 2부터 5까지 17개 칸으로 나눔.
        list_size = np.size(list[idx])
        len_list.append(list_size)          # list_size는 총 데이터의 개수
        # 나중에 엔트로피 구할 때, logp(x)를 구해야하는데 p(x)가 0이 되어버리면 logp(x)값이 무한이 되어버려서 에러가 난다. 그래서 0에 가까운 수로 만들어 주기 위해 추가.
        hist = hist + np.ones(16)
        prob = hist/(list_size+16)
        all_prob.append(prob)

    bin_center = 0.5 * (bin_edge[1:] + bin_edge[:-1])

    # plt.figure()
    # plt.plot(bin_center, hist)
    # plt.show()

    return all_prob, len_list


def smoothListGaussian(list,degree=5):

    window=degree*2-1 # 9
    weight=np.array([1.0]*window)
    weightGauss=[]
    for i in range(window):
        i=i-degree+1 #
        frac=i/float(window)
        gauss=1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight=np.array(weightGauss)*weight
    smoothed=[0.0]*(len(list)-window)
    for i in range(len(smoothed)):
        smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)
    return smoothed

if __name__ == "__main__":
    # train_battery_list = ['5_E_Battery', '6_F_Battery', '7_G_Battery']
    # dir_path = "../data/battery_device/Data_Excel_0910"
    #
    # voltage_list = []
    #
    # for battery in train_battery_list:
    #     BatteryDataDir = os.path.join(dir_path, battery)
    #     FileNameList = os.listdir(BatteryDataDir)
    #
    #     D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle, DC_list_time_all = ReadData_random(FileNameList, BatteryDataDir)
    #
    #     np.save(BatteryDataDir + '/capacity', Capacity)
    #     np.save(BatteryDataDir + '/SOH', SOH)
    #     np.save(BatteryDataDir + '/discharge_data', D_Voltage)
    #     np.save(BatteryDataDir + '/discharge_current', D_Current)
    #     np.save(BatteryDataDir + '/charge_data', C_Voltage)
    #     np.save(BatteryDataDir + '/charge_current', C_Current)
    #     np.save(BatteryDataDir + '/last_cycle', last_cycle)
    #     np.save(BatteryDataDir + '/discharge_time_all', DC_list_time_all)


    #train_battery_list = ['1_A_Battery', '2_B_Battery', '8_H_Battery']
    #train_battery_list = ['7_G_Battery']  #[ '7_G_Battery']   # , '6_F_Battery', '7_G_Battery'
    train_battery_list = ['Charge_data']
    dir_path = "../data/battery_device/Data_Excel_ChargeData"

    voltage_list = []

    for battery in train_battery_list:
        BatteryDataDir = os.path.join(dir_path, battery)
        FileNameList = os.listdir(BatteryDataDir)

        #Capacity, SOH = ReadData_capapcity(FileNameList, BatteryDataDir)
        #ReadData_constant
        #D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle, DC_list_time_all, C_list_time_all = ReadData_random(FileNameList, BatteryDataDir)
        D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle, DC_list_time_all = ReadData_constant(
            FileNameList, BatteryDataDir)

        plt.figure()
        plt.plot(Capacity)
        plt.show()

        plt.figure()
        plt.plot(SOH)
        plt.show()


        # np.save(BatteryDataDir + '/capacity2', Capacity)
        # np.save(BatteryDataDir + '/SOH2', SOH)
        np.save(BatteryDataDir + '/capacity', Capacity)
        np.save(BatteryDataDir + '/SOH', SOH)
        np.save(BatteryDataDir + '/discharge_data', D_Voltage)
        np.save(BatteryDataDir + '/discharge_current', D_Current)
        np.save(BatteryDataDir + '/charge_data', C_Voltage)
        np.save(BatteryDataDir + '/charge_current', C_Current)
        # np.save(BatteryDataDir + '/last_cycle', last_cycle)
        np.save(BatteryDataDir + '/discharge_time_all', DC_list_time_all)
#        np.save(BatteryDataDir + '/charge_time_all', C_list_time_all)





