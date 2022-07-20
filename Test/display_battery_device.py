import numpy as np
import matplotlib.pyplot as plt
from utils.Entropy import Moving_Avg_Filter
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import operator
from scipy import integrate


## 5E mean 1.4, var 0.45
## 6F mean 2.2, var 1.0
## 7G mean 1.6, var 0.9

def display_cap_entropy_discharge_device():
    check_capacity1 = np.load('../data/battery_device/Data_Excel_210320/1_A_Battery/capacity.npy', allow_pickle=True)
    check_capacity2 = np.load('../data/battery_device/Data_Excel_210320/8_H_Battery/capacity.npy', allow_pickle=True)
    check_capacity5 = np.load('../data/battery_device/Data_Excel_210320/5_E_Battery/capacity.npy', allow_pickle=True)
    check_capacity6 = np.load('../data/battery_device/Data_Excel_210320/6_F_Battery/capacity.npy', allow_pickle=True)
    check_capacity7 = np.load('../data/battery_device/Data_Excel_210320/7_G_Battery/capacity.npy', allow_pickle=True)

    check_constant_current1 = np.load('../data/battery_device/Data_Excel_210320/1_A_Battery/discharge_current.npy', allow_pickle=True)
    check_constant_current2 = np.load('../data/battery_device/Data_Excel_210320/8_H_Battery/discharge_current.npy', allow_pickle=True)
    check_constant_current5 = np.load('../data/battery_device/Data_Excel_210320/5_E_Battery/discharge_current.npy', allow_pickle=True)
    check_constant_current6 = np.load('../data/battery_device/Data_Excel_210320/6_F_Battery/discharge_current.npy', allow_pickle=True)
    check_constant_current7 = np.load('../data/battery_device/Data_Excel_210320/7_G_Battery/discharge_current.npy', allow_pickle=True)

    check_constant_entropy1 = np.load('../data/battery_device/Data_Excel_210320/1_A_Battery/discharge_Entropy_reverse.npy', allow_pickle=True)
    check_constant_entropy2 = np.load('../data/battery_device/Data_Excel_210320/8_H_Battery/discharge_Entropy_reverse.npy', allow_pickle=True)
    check_constant_entropy5 = np.load('../data/battery_device/Data_Excel_210320/5_E_Battery/discharge_Entropy_reverse.npy', allow_pickle=True)
    check_constant_entropy6 = np.load('../data/battery_device/Data_Excel_210320/6_F_Battery/discharge_Entropy_reverse.npy', allow_pickle=True)
    check_constant_entropy7 = np.load('../data/battery_device/Data_Excel_210320/7_G_Battery/discharge_Entropy_reverse.npy', allow_pickle=True)

    plt.figure()
    plt.plot(check_constant_entropy2)
    plt.show()

    check_constant_charge1 = np.load('../data/battery_device/Data_Excel_210320/1_A_Battery/discharge_Charge_2.npy', allow_pickle=True)     #discharge_Entropy
    check_constant_charge2 = np.load('../data/battery_device/Data_Excel_210320/8_H_Battery/discharge_Charge_2.npy', allow_pickle=True)
    check_constant_charge5 = np.load('../data/battery_device/Data_Excel_210320/5_E_Battery/discharge_Charge_2.npy', allow_pickle=True)
    check_constant_charge6 = np.load('../data/battery_device/Data_Excel_210320/6_F_Battery/discharge_Charge_2.npy', allow_pickle=True)
    check_constant_charge7 = np.load('../data/battery_device/Data_Excel_210320/7_G_Battery/discharge_Charge_2.npy', allow_pickle=True)

    # check_constant_time = np.load('../data/battery_device/Data_Excel_1224/8_H_Battery/discharge_time_all.npy', allow_pickle=True)
    # check_constant_voltage = np.load('../data/battery_device/Data_Excel_1224/1_A_Battery/discharge_data.npy', allow_pickle=True)

    current_mean1 = []
    current_var1 = []

    current_mean2 = []
    current_var2 = []

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
    check_SOH5 = check_capacity5/3.25
    check_SOH6 = check_capacity6/3.25
    check_SOH7 = check_capacity7/3.25

    for i in range(len(check_SOH1)):
        check_SOH1[i] = round(check_SOH1[i], 3)
    for i in range(len(check_SOH2)):
        check_SOH2[i] = round(check_SOH2[i], 3)
    for i in range(len(check_SOH5)):
        check_SOH5[i] = round(check_SOH5[i], 3)
    for i in range(len(check_SOH6)):
        check_SOH6[i] = round(check_SOH6[i], 3)
    for i in range(len(check_SOH7)):
        check_SOH7[i] = round(check_SOH7[i], 3)

    SOH1_ent_dic = {}
    SOH2_ent_dic = {}
    SOH5_ent_dic = {}
    SOH6_ent_dic = {}
    SOH7_ent_dic = {}

    SOH1_charge_dic = {}
    SOH2_charge_dic = {}
    SOH5_charge_dic = {}
    SOH6_charge_dic = {}
    SOH7_charge_dic = {}

    for i in range(len(check_SOH1)):
        try:
            if SOH1_ent_dic.get(check_SOH1[i]) != None:
                SOH1_ent_dic[check_SOH1[i]] = (SOH1_ent_dic.get(check_SOH1[i]) + check_constant_entropy1[i]) / 2
                SOH1_charge_dic[check_SOH1[i]] = (SOH1_charge_dic.get(check_SOH1[i]) + (check_constant_charge1[i]/11520)) / 2
            else:
                SOH1_ent_dic[check_SOH1[i]] = check_constant_entropy1[i]
                SOH1_charge_dic[check_SOH1[i]] = check_constant_charge1[i]/11520
        except:
            continue

    for i in range(len(check_SOH2)):
        try:
            if SOH2_ent_dic.get(check_SOH2[i]) != None:
                SOH2_ent_dic[check_SOH2[i]] = (SOH2_ent_dic.get(check_SOH2[i]) + check_constant_entropy2[i]) / 2
                SOH2_charge_dic[check_SOH2[i]] = (SOH2_charge_dic.get(check_SOH2[i]) + (check_constant_charge2[i]/11520)) / 2
            else:
                SOH2_ent_dic[check_SOH2[i]] = check_constant_entropy2[i]
                SOH2_charge_dic[check_SOH2[i]] = check_constant_charge2[i]/11520
        except:
            continue

    for i in range(len(check_SOH5)):
        try:
            if SOH5_ent_dic.get(check_SOH5[i]) != None:
                SOH5_ent_dic[check_SOH5[i]] = (SOH5_ent_dic.get(check_SOH5[i]) + check_constant_entropy5[i]) / 2
                SOH5_charge_dic[check_SOH5[i]] = (SOH5_charge_dic.get(check_SOH5[i]) + (check_constant_charge5[i]/11520)) / 2
            else:
                SOH5_ent_dic[check_SOH5[i]] = check_constant_entropy5[i]
                SOH5_charge_dic[check_SOH5[i]] = check_constant_charge5[i]/11520
        except:
            continue

    for i in range(len(check_SOH6)):
        if SOH6_ent_dic.get(check_SOH6[i]) != None:
            SOH6_ent_dic[check_SOH6[i]] = (SOH6_ent_dic.get(check_SOH6[i]) + check_constant_entropy6[i]) / 2
            SOH6_charge_dic[check_SOH6[i]] = (SOH6_charge_dic.get(check_SOH6[i]) + (check_constant_charge6[i]/11520)) / 2
        else:
            SOH6_ent_dic[check_SOH6[i]] = check_constant_entropy6[i]
            SOH6_charge_dic[check_SOH6[i]] = check_constant_charge6[i]/11520
        # except:
        #     continue

    for i in range(len(check_SOH7)):
        try:
            if SOH7_ent_dic.get(check_SOH7[i]) != None:
                SOH7_ent_dic[check_SOH7[i]] = (SOH7_ent_dic.get(check_SOH7[i]) + check_constant_entropy7[i]) / 2
                SOH7_charge_dic[check_SOH7[i]] = (SOH7_charge_dic.get(check_SOH7[i]) + (check_constant_charge7[i]/11520)) / 2
            else:
                SOH7_ent_dic[check_SOH7[i]] = check_constant_entropy7[i]
                SOH7_charge_dic[check_SOH7[i]] = check_constant_charge7[i]/11520
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
    charge_1_2 = []
    for i in range(len(check_SOH1)):
        entropy1_2.append(SOH1_ent_dic[check_SOH1[i]])
        charge_1_2.append(SOH1_charge_dic[check_SOH1[i]])

    entropy2_2 = []
    charge_2_2 = []
    for i in range(len(check_SOH2)):
        entropy2_2.append(SOH2_ent_dic[check_SOH2[i]])
        charge_2_2.append(SOH2_charge_dic[check_SOH2[i]])

    entropy5_2 = []
    charge_5_2 = []
    for i in range(len(check_SOH5)):
        entropy5_2.append(SOH5_ent_dic[check_SOH5[i]])
        charge_5_2.append(SOH5_charge_dic[check_SOH5[i]])

    entropy6_2 = []
    charge_6_2 = []
    for i in range(len(check_SOH6)):
        entropy6_2.append(SOH6_ent_dic[check_SOH6[i]])
        charge_6_2.append(SOH6_charge_dic[check_SOH6[i]])

    entropy7_2 = []
    charge_7_2 = []
    for i in range(len(check_SOH7)):
        entropy7_2.append(SOH7_ent_dic[check_SOH7[i]])
        charge_7_2.append(SOH7_charge_dic[check_SOH7[i]])


    '''
    Fig. 11.  The Change of Voltage entropy according to Aging (SoH) parameterized by Average and Variance of Discharging Current.  
    '''
    plt.figure()
    plt.plot(check_SOH1[2:], entropy1_2[2:], label='Battery 1, m=1.6A, s=0')
    plt.plot(check_SOH2[2:-10], entropy2_2[2:-10], label='Battery 2, m=3.2A, s=0')
    plt.plot(check_SOH5[:-6], entropy5_2[:-6], label='Battery 3, m=1.4A, s=0.46')
    plt.plot(check_SOH6[23:], entropy6_2[23:], label='Battery 4, m=2.2A, s=1.02')
    plt.plot(check_SOH7[2:], entropy7_2[2:], label='Battery 5, m=1.6A, s=0.92')
    plt.ylabel('Entropy')
    plt.xlabel('SOH (%)')
    plt.legend()
    plt.gca().invert_xaxis()
    #plt.ylim(0.54, 1.5)
    #plt.show()

    '''
    Fig. 13. The Change of Discharging Volume according to Aging (SoH) parameterized by Average and Variance of Discharging Current.
    '''
    plt.figure()
    plt.plot(check_SOH1[2:], charge_1_2[2:], label='Battery 1, m=1.6A, s=0')
    plt.plot(check_SOH2[2:-10], charge_2_2[2:-10], label='Battery 2, m=3.2A, s=0')
    plt.plot(check_SOH5[:-6], charge_5_2[:-6], label='Battery 3, m=1.4A, s=0.46')
    plt.plot(check_SOH6[23:], charge_6_2[23:], label='Battery 4, m=2.2A, s=1.02')
    plt.plot(check_SOH7[2:], charge_7_2[2:], label='Battery 5, m=1.6A, s=0.92')
    plt.ylabel('Charge Value')
    plt.xlabel('SOH (%)')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()



if __name__ == "__main__":
    display_cap_entropy_discharge_device()