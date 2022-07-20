import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def AutoCorr(list, lag):
    AC_list = []
    correlation = np.corrcoef(list, list)
    correlation = correlation[0][1]
    AC_list.append(correlation)
    for shift in range(1, lag+1):
        list_lag = list[shift:]
        list_lag_minus = list[:-shift]
        correlation = np.corrcoef(list_lag, list_lag_minus)[0][1]
        AC_list.append(correlation)

    return AC_list

def Entropy0(density):
    ''' normal Entropy'''
    # x = np.multiply(x,100)
    # indices = x - 200
    # density_list = [density[int(x)] for x in indices]
    ent = -1 * np.sum(np.multiply(density, np.log10(density)))
    # ent = -1*ent1/np.log2(len(density))

    return ent

def Entropy1(density, len_value,current_value, beta):
    ''' time compensated Entropy'''
    abs_current = np.abs(current_value)

    # current = 0
    #
    # for i in range(3, len(abs_current)):
    #     if abs_current[i] > 0.1:
    #         current = abs_current[i]
    #         break
    #
    # time_compenate = len_value[-1]

    integ_CB = integrate.cumtrapz(abs_current, len_value)[-1]

    # x = np.multiply(x,100)
    # indices = x - 200
    # density_list = [density[int(x)] for x in indices]
    ent1 = np.sum(np.multiply(density, np.log10(density)))
    # ent = -1*ent1/np.log2(len(density))
    # ent = -1*beta*ent1/(current * time_compenate)
    ent = -1 * beta * ent1 / integ_CB

    return ent


def Entropy2(density, len_value, current_value, beta):
    '''current constant'''
    abs_current = np.abs(current_value)

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    time_compenate = len_value[-1]

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    # if len_value == 0:
    #     print("len_val is zero")
    # if current == 0:
    #     current = 0.001
    #     print("current is zero ")


    ent = -1 * beta * ent1 * (1.041 / np.exp(0.0214 * current)) / (current*time_compenate)

    return ent, np.abs(np.round(current, 2))

def Entropy2_2(density, len_value, current_value, beta, time_rate):
    '''current changing'''
    abs_current = np.abs(current_value)

    # current = 2
    # last_time = 0

    # for i in range(3, len(time_rate)-1):
    #     if (time_rate[i+1]-time_rate[i]) > 0:
    #         current = current+(abs_current[i]*(time_rate[i+1]-time_rate[i]))
    #         last_time = last_time + (time_rate[i+1]-time_rate[i])
    #
    # current = current/last_time

    current = np.average(abs_current)

    integ_CB = integrate.cumtrapz(abs_current, time_rate)[-1]

    # integ_CB = 0
    #
    # for i in range(1, len(time_rate)-1):
    #     dx1 = (time_rate[i] + time_rate[i-1]) /2
    #     dx2 = (time_rate[i+1] + time_rate[i]) /2
    #
    #     integ_CB = integ_CB + ( -current_value[i]*(dx2 - dx1))

    # current = integ_CB/time_rate[-1]

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    ent = -1 * beta * ent1 * (1.05 / np.exp(0.0278 * current)) / (integ_CB)  #(current * len_value)  #

    return ent, np.abs(np.round(current, 2))

def Entropy2_2_prev(density, len_value, current_value, beta, rate, mode, no_coff):
    '''current changing'''
    if mode is 'discharge':
        min_current = np.abs(np.round(np.min(current_value),2))
        if min_current > 2.3:
            abs_min_current = 1
        else:
            abs_min_current = min_current
        ent1 = np.sum(np.multiply(density, np.log10(density)))
        if no_coff: # coefficient 없는 경우
            ent = -1 * beta * ent1 / (abs_min_current * len_value * rate)
        else:
            ent = -1 * beta * ent1 * (1.31 / np.exp(0.227 * abs_min_current)) / (abs_min_current * len_value * rate)

        return ent, min_current
    elif mode is 'charge':
        charge_current = np.abs(np.round(np.max(current_value), 2))
        ent1 = np.sum(np.multiply(density, np.log10(density)))
        ent = -1 * beta * ent1 * (1.31 / np.exp(0.227 * 1))  / (1*len_value)

        return ent, charge_current

    else:
        print('please select discharge and charge mode')
        exit()

def Entropy3(density, len_value, current_value, beta, mode='compensated'):
    '''compensate or not 비교하기위한 함수'''
    min_current = np.min(current_value)
    if min_current > -0.2:
        abs_min_current = 1
    else:
        abs_min_current = np.abs(min_current)
    ent1 = np.sum(np.multiply(density, np.log10(density)))
    if mode is 'compensated':
        ent = -1*beta*ent1*(1.31 / np.exp(0.227*abs_min_current)) /(abs_min_current * len_value)
        # ent = -1*beta*ent1/(abs_min_current * len_value)
    elif mode is 'not':
        print("not compensated enropy")
        ent = -1*ent1
    else:
        print("mode should be 'compensated' or 'not'.")
        raise AssertionError

    # ent = -1*beta*ent1/(np.log(len_value*abs_min_current)/3)
    # ent = -1*beta*ent1/len_value

    return ent, np.round(min_current, 2)

def Entropy4_DC(density, len_value, current_value, beta, rated_cap):
    '''current constant, check Rated Capacity'''
    abs_current = np.abs(current_value)

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    ent2 = -1 * beta * ent1 * (1.31 / np.exp(0.227 * current)) / (current*len_value)

    ent = ent2 * 0.352 * np.exp(0.934 * rated_cap)

    #ent = ent2

    return ent, np.abs(np.round(current, 2))

def Entropy4_C(density, len_value, current_value, beta, rated_cap):
    '''current constant, check Rated Capacity'''
    abs_current = np.abs(current_value)

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    ent2 = -1 * beta * ent1 * (1.31 / np.exp(0.227 * current)) / (current*len_value)

    ent = ent2 * 0.637 * np.exp(0.43 * rated_cap)


    #ent = ent2

    return ent, np.abs(np.round(current, 2))

def Moving_Avg_Filter(list, num):
    '''moving Average filter'''
    new_list = []

    for i in range(len(list)):
        if i < num:
            sum = 0
            for j in range(i+1):
                sum = sum + list[j]
            new_list.append(sum/(i+1))
        else:
            sum = 0
            for j in range(i-num+1, i+1):
                sum = sum + list[j]
            new_list.append(sum/num)

    return new_list

def MF(list, kernal_size):
    '''median filter'''
    new_list = [np.median(list[idx:idx+kernal_size]) for idx in range(len(list) - kernal_size+1)]
    for _ in range(kernal_size-1):
        new_list.append(np.median(list[len(list)-kernal_size:len(list)]))

    return new_list

def drawEntropy(density, len_list):
    entropy_list = []
    min_value = min(len_list)
    for idx in range(len(density)):
        entropy = Entropy3(density[idx], len_list[idx], 50)
        entropy_list.append(entropy)
    return entropy_list

def Density(list):
    # all_exp_log_dens = []
    all_prob = []
    len_list = []
    for idx in range(len(list)):
        hist, bin_edges = np.histogram(list[idx], bins=np.linspace(2,5,17))
        list_size = len(list[idx])
        len_list.append(list_size)
        hist = hist + np.ones(16)
        prob = hist/(list_size+16)

        all_prob.append(prob)

    return all_prob, len_list

def EntropyForSOH(list, beta):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    all_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list)):
        prob_list = []
        temp_len = []
        for idx2 in range(len(list[idx])): #cycle
            hist, bin_edges = np.histogram(list[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = len(list[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))
        len_list.append(temp_len)
        all_prob.append(prob_list)

    all_entropy_list = []
    for idx in range(len(list)):
        entropy_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list[idx])):
            entropy_list.append(Entropy3(all_prob[idx][idx2], len_list[idx][idx2], beta))
        # entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)

    return all_entropy_list

def EntropyAndProb(list, beta):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    all_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list)):
        prob_list = []
        temp_len = []
        for idx2 in range(len(list[idx])): #cycle
            hist, bin_edges = np.histogram(list[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = len(list[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))
        len_list.append(temp_len)
        all_prob.append(prob_list)

    all_entropy_list = []
    for idx in range(len(list)):
        entropy_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list[idx])):
            # idx 번째 배터리의 idx2번째 cycle
            entropy_list.append(Entropy3(all_prob[idx][idx2], len_list[idx][idx2], beta))
        entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)

    return all_entropy_list, all_prob

def DistEntropyForSOH(M, list):
    '''
    Distribution Entropy 계산
    :param M: the number of bins
    :param list:
    :return:
    '''
    all_prob_list = []
    for idx in range(len(list)):
        prob_list = []
        for idx2 in range(len(list[idx])):
            UDM = UpperDM(list[idx][idx2])
            hist, bin_edges = np.histogram(UDM, bins=np.linspace(0,2, M+1))
            hist = hist + np.ones(M)
            list_size = len(UDM) + M
            prob_list.append(hist/list_size)
        all_prob_list.append(prob_list)
    all_entropy_list = []
    for idx in range(len(list)):
        entropy_list = []
        for idx2 in range(len(list[idx])):
            entropy_list.append(-1*np.sum(all_prob_list[idx][idx2]*np.log2(all_prob_list[idx][idx2]))/np.log2(M))
        entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)

    return all_entropy_list

def EntropyForSOHProb_withCurrent_oneNasaBattery(list_A, current_list, beta, mode, time, rate, SOH):
    '''
    기존의 EntropyForSOHProb_withCurrent는 전체 battery data를 한번에 처리한다.
    이 Function은 1개의 battery data만 처리한다.
    :param list_A: 전압 정보, [cycle][voltage]
    :param current_list: 전류정보, [cycle][curent]
    :param beta: lenght coefficient
    :param mode: discharge/charge mode
    :return entropy_list: entropy [cycle][entropy]
    :return retrun_prob_list : probability [cycle][probability]
    '''
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''

    prob_list = []
    list_len = []
    retrun_prob_list = []

    Cal_Ri = []

    for idx2 in range(len(list_A)): #cycle
        if len(list_A[idx2]) == 0:
            continue

        tmp_Cal_Ri = []
        for i in range(1, len(list_A[idx2]-1)):
            vol_diff = list_A[idx2][i-1] - list_A[idx2][i]
            vol_diff = np.abs(vol_diff)
            tmp_curr = np.abs(current_list[idx2][i-1])
            if tmp_curr > 0.01:
                tmp_Cal_Ri.append(vol_diff / tmp_curr)
                #print(" i : ", i, "voltage1 : ", list_A[idx2][i], "voltage2 : ", list_A[idx2][i - 1], "current : ",
                 #     -current_list[idx2][i], " vol_diff : ", vol_diff, "Ri : ", tmp_Cal_Ri[-1])

        Cal = np.average(tmp_Cal_Ri)
        if Cal > 0.15:
            Cal = Cal_Ri[-1]
        Cal_Ri.append(Cal)

        #Ri = (-0.01713 * SOH[idx2]) + 0.1198

        Ri = (-0.0152 * SOH[idx2]) + 0.134

        # V = IR
        for i in range(len(list_A[idx2])):
            # print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx2][i], "resistance : ", Ri)
            list_A[idx2][i] = list_A[idx2][i] + (Ri*(-current_list[idx2][i]))

        hist, bin_edges = np.histogram(list_A[idx2], bins=np.linspace(2,5,17)) # 16 bins
        list_size = len(list_A[idx2])
        list_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
        hist = hist + np.ones(16)
        prob_list.append(hist/(list_size+16))

        hist, _ = np.histogram(list_A[idx2], bins=np.linspace(2, 5, 7))
        #retrun_prob_list.append(hist/list_size)

    entropy_list = []

    # min_len_value = min(len_list[idx]) # 길이의 최소값

    for idx2 in range(1, len(list_A)-1):
        # try:
        if len(list_A[idx2]) == 0:
            continue
        #entropy_cycle = Entropy0(prob_list[idx2])

        #entropy_cycle = Entropy1(prob_list[idx2], rate[idx2], current_list[idx2], beta)

        #entropy_cycle, min_current = Entropy2(prob_list[idx2], rate[idx2], current_list[idx2], beta)

        entropy_cycle, min_current = Entropy2_2(prob_list[idx2], time[idx2], current_list[idx2], beta, rate[idx2])

        # #entropy_cycle = Entropy3(prob_list[idx2], list_len[idx2], beta)

        # if mode is 'discharge':
        #     entropy_cycle, min_current = Entropy4_DC(prob_list[idx2], time[idx2], current_list[idx2], beta, ratedCap)
        # else:
        #     entropy_cycle, min_current = Entropy4_C(prob_list[idx2], time[idx2],
        #                                              current_list[idx2], beta, ratedCap)

        entropy_list.append(entropy_cycle)
        # except:
        #     print("Exception idx : ", idx2)
    entropy_list = MF(entropy_list, 3)

    if mode is 'discharge':
        return entropy_list, prob_list, list_A
    elif mode is 'charge':
        return entropy_list, retrun_prob_list


def EntropyForSOHProb_total_cur_realvehicle(list_A, current_list, mode, SOH, All_DC_time):
    list_len = []
    retrun_prob_list = []

    all_list = []
    all_cap = []
    all_cur_list = []


    for idx in range(len(list_A)):  # cycle
        prob_list = []

        entropy_list = []
        curremt_sum_list = []

        entropy_cur_list = []

        for idx2 in range(len(list_A[idx])):  # cycle
            if len(list_A[idx][idx2]) <= 1:
                if idx2 == 0:
                    entropy_list.append(0)
                else:
                    entropy_list.append(entropy_list[-1])
                continue

            # INR 18650 Battery Data
            Ri = (96 / 32) * ((-0.4722347 * SOH[idx][idx2]) + 0.62305638)  # 62305638

            for i in range(len(list_A[idx][idx2])):
                try:
                    list_A[idx][idx2][i] = list_A[idx][idx2][i] + (np.multiply(Ri, np.abs(current_list[idx][idx2][i])))
                except:
                    print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx][idx2][i], "resistance : ", Ri)

            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(350, 400, 1000))  # 16 bins    prev 17
            list_size = len(list_A[idx][idx2])
            list_len.append(list_size)  # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(999)  # prev 16

            prob_list = hist / (list_size + 999)  # prev 16

            current_min = np.min(np.abs(current_list[idx][idx2])) - 10
            current_max = np.max(np.abs(current_list[idx][idx2])) + 10


            hist_cur, bin_edges_cur = np.histogram(np.abs(current_list[idx][idx2]), bins=np.linspace(current_min, current_max, 100))  # 16 bins    prev 17
            list_size_cur = len(current_list[idx][idx2])
            hist_cur = hist_cur + np.ones(99)

            prob_list_cur = hist_cur / (list_size_cur + 99)

            try:
                entropy_cycle = Entropy0(prob_list)
                entropy_cycle_cur = Entropy0(prob_list_cur)
            except:
                entropy_cycle = entropy_list[-1]
                entropy_cycle_cur = entropy_cycle_cur[-1]

            entropy_list.append(entropy_cycle)
            entropy_cur_list.append(entropy_cycle_cur)

        check_curremt_sum = []
        for i in range(len(All_DC_time[idx])):
            if len(current_list[idx][i]) <= 1 or len(All_DC_time[idx][i]) == 0 :
                #print("list size is zero, idx : ", i)
                if i == 0:
                    check_curremt_sum.append(0)
                else:
                    check_curremt_sum.append(check_curremt_sum[-1])
                continue

            current = np.abs(current_list[idx][i][0])
            try:
                integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i]), All_DC_time[idx][i])[-1]
                check_curremt_sum.append(integ_CB)
                #check_curremt_sum.append((((All_DC_time[idx][i][-1]) * current) + (223.618 * np.exp(0.977 * current)) - 259.77) / 100)
            except:
                print("All_DC_time[idx][i] :", All_DC_time[idx][i], " All_DC_time[idx][i][0] : ", All_DC_time[idx][i])
                #print("elapsed_time : ", elapsed_time, " current : ", current)

        check_curremt_sum = Moving_Avg_Filter(check_curremt_sum[:], 5)
        entropy_list = Moving_Avg_Filter(entropy_list[:], 5)
        #entropy_cur_list = Moving_Avg_Filter(entropy_cur_list[:], 5)

        ret_mul = []

        # for i in range(len(entropy_list)):
        #     entropy_list[i] = entropy_list[i] * 4300
        #     ret_mul.append((((entropy_list[i] ** (current)) * ((4200-check_curremt_sum[i]) ** (2.3-(current))))**(1/2.3))-(932.63*(0.11**(1.31))-1000))

        all_list.append(entropy_list)
        all_cap.append(check_curremt_sum)
        all_cur_list.append(entropy_cur_list)

    if mode is 'discharge':
        return all_list, all_cap, prob_list, list_A, curremt_sum_list, ret_mul, all_cur_list
    elif mode is 'charge':
        return all_list, retrun_prob_list


def EntropyForSOHProb_total_cur(list_A, current_list, mode, SOH, All_DC_time):
    list_len = []
    retrun_prob_list = []

    all_list = []
    all_cap = []
    all_cur_list = []


    for idx in range(len(list_A)):  # cycle
        prob_list = []

        entropy_list = []
        curremt_sum_list = []

        entropy_cur_list = []

        for idx2 in range(len(list_A[idx])):  # cycle
            if len(list_A[idx][idx2]) <= 1:
                if idx2 == 0:
                    entropy_list.append(0)
                else:
                    entropy_list.append(entropy_list[-1])
                continue

            # INR 18650 Battery Data
            #Ri =(-0.0357 * SOH[idx][idx2]) + 0.19  #0.15

            Ri = (-0.4722347 * SOH[idx][idx2]) + 0.62305638  # 62305638

            for i in range(len(list_A[idx][idx2])):
                try:
                    list_A[idx][idx2][i] = list_A[idx][idx2][i] + (np.multiply(Ri, np.abs(current_list[idx][idx2][i])))
                except:
                    print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx][idx2][i], "resistance : ", Ri)

            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(3.4, 4.2, 17))  # 16 bins    prev 17
            list_size = len(list_A[idx][idx2])
            list_len.append(list_size)  # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)  # prev 16

            prob_list = hist / (list_size + 16)  # prev 16

            current_min = np.min(np.abs(current_list[idx][idx2])) - 0.3
            current_max = np.max(np.abs(current_list[idx][idx2])) + 0.3


            hist_cur, bin_edges_cur = np.histogram(np.abs(current_list[idx][idx2]), bins=np.linspace(current_min, current_max, 10))  # 16 bins    prev 17
            list_size_cur = len(current_list[idx][idx2])
            hist_cur = hist_cur + np.ones(9)

            prob_list_cur = hist_cur / (list_size_cur + 9)

            try:
                entropy_cycle = Entropy0(prob_list)
                entropy_cycle_cur = Entropy0(prob_list_cur)
            except:
                entropy_cycle = entropy_list[-1]
                entropy_cycle_cur = entropy_cycle_cur[-1]

            entropy_list.append(entropy_cycle)
            entropy_cur_list.append(entropy_cycle_cur)

        check_curremt_sum = []
        for i in range(len(All_DC_time[idx])):
            if len(current_list[idx][i]) <= 1 or len(All_DC_time[idx][i]) == 0 :
                #print("list size is zero, idx : ", i)
                if i == 0:
                    check_curremt_sum.append(0)
                else:
                    check_curremt_sum.append(check_curremt_sum[-1])
                continue

            current = np.abs(current_list[idx][i][0])
            try:
                if current_list[idx][i][0] < -3.3:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i])+0.2, All_DC_time[idx][i])[-1]
                else:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i]), All_DC_time[idx][i])[-1]
                check_curremt_sum.append(integ_CB)
                #check_curremt_sum.append((((All_DC_time[idx][i][-1]) * current) + (223.618 * np.exp(0.977 * current)) - 259.77) / 100)
            except:
                print("All_DC_time[idx][i] :", All_DC_time[idx][i], " All_DC_time[idx][i][0] : ", All_DC_time[idx][i])
                #print("elapsed_time : ", elapsed_time, " current : ", current)

        check_curremt_sum = Moving_Avg_Filter(check_curremt_sum[:], 5)
        entropy_list = Moving_Avg_Filter(entropy_list[:], 5)
        #entropy_cur_list = Moving_Avg_Filter(entropy_cur_list[:], 5)

        ret_mul = []

        # for i in range(len(entropy_list)):
        #     entropy_list[i] = entropy_list[i] * 4300
        #     ret_mul.append((((entropy_list[i] ** (current)) * ((4200-check_curremt_sum[i]) ** (2.3-(current))))**(1/2.3))-(932.63*(0.11**(1.31))-1000))

        all_list.append(entropy_list)
        all_cap.append(check_curremt_sum)
        all_cur_list.append(entropy_cur_list)

    if mode is 'discharge':
        return all_list, all_cap, prob_list, list_A, curremt_sum_list, ret_mul, all_cur_list
    elif mode is 'charge':
        return all_list, retrun_prob_list



def EntropyForSOHProb_total_cur_dtdv(list_A, current_list, mode, SOH, All_DC_time):
    list_len = []
    retrun_prob_list = []

    all_list = []
    all_cap = []
    all_cur_list = []


    for idx in range(len(list_A)):  # cycle
        prob_list = []

        entropy_list = []
        curremt_sum_list = []

        entropy_cur_list = []

        for idx2 in range(len(list_A[idx])):  # cycle
            if len(list_A[idx][idx2]) <= 1:
                if idx2 == 0:
                    entropy_list.append(0)
                else:
                    entropy_list.append(entropy_list[-1])
                continue

            # INR 18650 Battery Data
            Ri = (-0.4722347 * SOH[idx][idx2]) + 0.62305638  # 62305638

            for i in range(len(list_A[idx][idx2])):
                try:
                    list_A[idx][idx2][i] = list_A[idx][idx2][i] + (np.multiply(Ri, np.abs(current_list[idx][idx2][i])))
                except:
                    print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx][idx2][i], "resistance : ", Ri)


            time_list = []

            for ii in range(len(list_A[idx][idx2])):
                time_list.append(ii * 30.0)

            voltage_max = np.max(list_A[idx][idx2])
            voltage_min = np.min(list_A[idx][idx2])

            hist_range = voltage_max - voltage_min

            time_max = np.max(time_list)

            _, bin_edges = np.histogram(time_list, bins=np.linspace(0, time_max, 17))

            voltage_idx1 = 0

            hist = []
            for hi in range(1, len(bin_edges)):
                voltage_idx2 = np.where(time_list < bin_edges[hi])[0][-1]
                vol_length = list_A[idx][idx2][voltage_idx1] - list_A[idx][idx2][voltage_idx2]

                if vol_length < 0:
                    vol_length = -vol_length

            hist.append(vol_length / hist_range * 100)
            if hist[-1] == 1:
                hist[-1] = hist[-1] + 0.00001
            voltage_idx1 = voltage_idx2

            #hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(3.4, 4.2, 17))  # 16 bins    prev 17
            list_size = len(list_A[idx][idx2])
            list_len.append(list_size)  # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)  # prev 16

            prob_list = hist / (list_size + 16)  # prev 16

            current_min = np.min(np.abs(current_list[idx][idx2])) - 0.3
            current_max = np.max(np.abs(current_list[idx][idx2])) + 0.3


            hist_cur, bin_edges_cur = np.histogram(np.abs(current_list[idx][idx2]), bins=np.linspace(current_min, current_max, 10))  # 16 bins    prev 17
            list_size_cur = len(current_list[idx][idx2])
            hist_cur = hist_cur + np.ones(9)

            prob_list_cur = hist_cur / (list_size_cur + 9)

            try:
                entropy_cycle = Entropy0(prob_list)
                entropy_cycle_cur = Entropy0(prob_list_cur)
            except:
                entropy_cycle = entropy_list[-1]
                entropy_cycle_cur = entropy_cycle_cur[-1]

            entropy_list.append(entropy_cycle)
            entropy_cur_list.append(entropy_cycle_cur)

        check_curremt_sum = []
        for i in range(len(All_DC_time[idx])):
            if len(current_list[idx][i]) <= 1 or len(All_DC_time[idx][i]) == 0 :
                #print("list size is zero, idx : ", i)
                if i == 0:
                    check_curremt_sum.append(0)
                else:
                    check_curremt_sum.append(check_curremt_sum[-1])
                continue

            current = np.abs(current_list[idx][i][0])
            try:
                if current_list[idx][i][0] < -3.3:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i])+0.2, All_DC_time[idx][i])[-1]
                else:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i]), All_DC_time[idx][i])[-1]
                check_curremt_sum.append(integ_CB)
                #check_curremt_sum.append((((All_DC_time[idx][i][-1]) * current) + (223.618 * np.exp(0.977 * current)) - 259.77) / 100)
            except:
                print("All_DC_time[idx][i] :", All_DC_time[idx][i], " All_DC_time[idx][i][0] : ", All_DC_time[idx][i])
                #print("elapsed_time : ", elapsed_time, " current : ", current)

        # check_curremt_sum = Moving_Avg_Filter(check_curremt_sum[:], 5)
        # entropy_list = Moving_Avg_Filter(entropy_list[:], 5)
        #entropy_cur_list = Moving_Avg_Filter(entropy_cur_list[:], 5)

        ret_mul = []

        # for i in range(len(entropy_list)):
        #     entropy_list[i] = entropy_list[i] * 4300
        #     ret_mul.append((((entropy_list[i] ** (current)) * ((4200-check_curremt_sum[i]) ** (2.3-(current))))**(1/2.3))-(932.63*(0.11**(1.31))-1000))

        all_list.append(entropy_list)
        all_cap.append(check_curremt_sum)
        all_cur_list.append(entropy_cur_list)

    if mode is 'discharge':
        return all_list, all_cap, prob_list, list_A, curremt_sum_list, ret_mul, all_cur_list
    elif mode is 'charge':
        return all_list, retrun_prob_list



def EntropyForSOHProb_Battery_Device_dtdv(list_A, current_list, mode, SOH, All_DC_time, isAreaFix=False):
    list_len = []
    retrun_prob_list = []

    all_list = []
    all_cap = []
    all_cur_list = []


    for idx in range(len(list_A)):  # cycle
        prob_list = []
        entropy_list = []
        curremt_sum_list = []
        entropy_cur_list = []

        tmp_hist = []
        tmp_bin_edge = []

        for idx2 in range(len(list_A[idx])):  # cycle
            if len(list_A[idx][idx2]) <= 1:
                if idx2 == 0:
                    entropy_list.append(0)
                else:
                    entropy_list.append(entropy_list[-1])
                continue

            # INR 18650 Battery Data
            Ri = (-0.4722347 * SOH[idx][idx2]) + 0.62305638  # 62305638

            for i in range(len(list_A[idx][idx2])):
                try:
                    list_A[idx][idx2][i] = list_A[idx][idx2][i] + (np.multiply(Ri, np.abs(current_list[idx][idx2][i])))
                except:
                    print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx][idx2][i], "resistance : ", Ri)

            time_list = []

            for ii in range(len(list_A[idx][idx2])):
                time_list.append(ii * 30.0)

            voltage_max = np.max(list_A[idx][idx2])
            voltage_min = np.min(list_A[idx][idx2])

            hist_range = voltage_max - voltage_min

            time_max = np.max(time_list)

            num_bin = 17

            if isAreaFix == False:
                _, bin_edges = np.histogram(time_list, bins=np.linspace(0, time_max, num_bin))
            else:
                _, bin_edges = np.histogram(time_list, bins=np.linspace(0, 8000, 17))

            voltage_idx1 = 0

            hist = []
            for hi in range(1, len(bin_edges)):
                voltage_idx2 = np.where(time_list < bin_edges[hi])[0][-1]
                vol_length = list_A[idx][idx2][voltage_idx1] - list_A[idx][idx2][voltage_idx2]
                if vol_length < 0:
                    vol_length = -vol_length

                hist.append(vol_length/hist_range*100)
                if hist[-1] == 1:
                    hist[-1] = hist[-1] + 0.00001
                voltage_idx1 = voltage_idx2

            list_size = len(list_A[idx][idx2])
            list_len.append(list_size)  # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(num_bin-1)  # prev 16

            tmp_hist.append(hist)
            tmp_bin_edge.append(bin_edges)

            prob_list = hist / (list_size + (num_bin-1))  # prev 16

            current_min = np.min(np.abs(current_list[idx][idx2])) - 0.3
            current_max = np.max(np.abs(current_list[idx][idx2])) + 0.3

            hist_cur, bin_edges_cur = np.histogram(np.abs(current_list[idx][idx2]), bins=np.linspace(current_min, current_max, 10))  # 16 bins    prev 17
            list_size_cur = len(current_list[idx][idx2])
            hist_cur = hist_cur + np.ones(9)

            prob_list_cur = hist_cur / (list_size_cur + 9)

            try:
                entropy_cycle = Entropy0(prob_list)
                entropy_cycle_cur = Entropy0(prob_list_cur)
            except:
                print("Exception")
                entropy_cycle = entropy_list[-1]
                entropy_cycle_cur = entropy_cycle_cur[-1]

            entropy_list.append(entropy_cycle)
            entropy_cur_list.append(entropy_cycle_cur)

        tmp_bin = []
        for bi in range(len(tmp_bin_edge)):
            tmp_bin.append([(tmp_bin_edge[bi][t]+tmp_bin_edge[bi][t+1])/2 for t in range(len(tmp_bin_edge[bi])-1)])

        '''
        Plot Voltage Distribution Histrogram
        '''
        # plt.figure()
        # plt.plot(tmp_bin[0], tmp_hist[0], label='cycle=0')
        # plt.plot(tmp_bin[100], tmp_hist[100], label='cycle=100')
        # plt.plot(tmp_bin[300], tmp_hist[300], label='cycle=300')
        # plt.plot(tmp_bin[500], tmp_hist[500], label='cycle=500')
        # plt.plot(tmp_bin[700], tmp_hist[700], label='cycle=700')
        # plt.legend()
        # plt.ylabel('Frequency')
        # plt.xlabel('Time (s)')
        # plt.show()

        check_curremt_sum = []
        for i in range(len(All_DC_time[idx])):
            if len(current_list[idx][i]) <= 1 or len(All_DC_time[idx][i]) == 0 :
                #print("list size is zero, idx : ", i)
                if i == 0:
                    check_curremt_sum.append(0)
                else:
                    check_curremt_sum.append(check_curremt_sum[-1])
                continue

            current = np.abs(current_list[idx][i][0])
            try:
                if current_list[idx][i][0] < -3.3:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i])+0.2, All_DC_time[idx][i])[-1]
                else:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i]), All_DC_time[idx][i])[-1]
                check_curremt_sum.append(integ_CB)
                #check_curremt_sum.append((((All_DC_time[idx][i][-1]) * current) + (223.618 * np.exp(0.977 * current)) - 259.77) / 100)
            except:
                print("All_DC_time[idx][i] :", All_DC_time[idx][i], " All_DC_time[idx][i][0] : ", All_DC_time[idx][i])
                #print("elapsed_time : ", elapsed_time, " current : ", current)

        # check_curremt_sum = Moving_Avg_Filter(check_curremt_sum[:], 5)
        # entropy_list = Moving_Avg_Filter(entropy_list[:], 5)
        # entropy_cur_list = Moving_Avg_Filter(entropy_cur_list[:], 5)

        ret_mul = []

        all_list.append(entropy_list)
        all_cap.append(check_curremt_sum)
        all_cur_list.append(entropy_cur_list)

    if mode is 'discharge':
        return all_list, all_cap, prob_list, list_A, curremt_sum_list, ret_mul, all_cur_list
    elif mode is 'charge':
        return all_list, retrun_prob_list


def EntropyForSOHProb_Battery_Device_dvdt22(list_A, current_list, mode, SOH, All_DC_time, isAreaFix=False):
    list_len = []
    retrun_prob_list = []

    all_list = []
    all_cap = []
    all_cur_list = []


    for idx in range(len(list_A)):  # cycle
        prob_list = []
        entropy_list = []
        curremt_sum_list = []
        entropy_cur_list = []

        tmp_hist = []
        tmp_bin_edge = []

        for idx2 in range(len(list_A[idx])):  # cycle
            if len(list_A[idx][idx2]) <= 1:
                if idx2 == 0:
                    entropy_list.append(0)
                else:
                    entropy_list.append(entropy_list[-1])
                continue

            # INR 18650 Battery Data
            Ri = (-0.4722347 * SOH[idx][idx2]) + 0.62305638  # 62305638   #4722347

            for i in range(len(list_A[idx][idx2])):
                try:
                    list_A[idx][idx2][i] = list_A[idx][idx2][i] + (np.multiply(Ri, np.abs(current_list[idx][idx2][i])))
                except:
                    print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx][idx2][i], "resistance : ", Ri)

            time_list = []

            for ii in range(len(list_A[idx][idx2])):
                time_list.append(ii * 30.0)

            voltage_max =  list_A[idx][idx2][0] # np.max(list_A[idx][idx2])
            voltage_min = list_A[idx][idx2][-1] #np.min(list_A[idx][idx2])

            hist_range = voltage_max - voltage_min

            time_max = np.max(time_list)

            num_bin = 17

            if isAreaFix == False:
                _, bin_edges = np.histogram(time_list, bins=np.linspace(0, time_max, num_bin))
            else:
                _, bin_edges = np.histogram(time_list, bins=np.linspace(0, 8000, 17))

            voltage_idx1 = 0


            hist = []
            for hi in range(1, len(bin_edges)):
                voltage_idx2 = np.where(time_list <= bin_edges[hi])[0][-1]
                vol_length = list_A[idx][idx2][voltage_idx1] - list_A[idx][idx2][voltage_idx2]
                if vol_length < 0:
                    hist.append(-vol_length/hist_range)
                    # hist[-1] = hist[-1] - 0.0000000001
                    # hist.append(0.0000000001)
                else:
                    hist.append(vol_length / hist_range)
                    voltage_idx1 = voltage_idx2
                    # # print("check list voltage_idx1 : ", voltage_idx1, " voltage_idx2 : ", voltage_idx2)
                    # print("CHekc vol : ", list_A[idx][idx2][voltage_idx1:voltage_idx2])
                    # vol_length = 0

            for hi in range(len(hist)):
                if hist[hi] > 1 or hist[hi] <= 0:
                    hist[hi] = 0.0000000001
                    print("prob more than 1 or less than 0 : ", hist[hi])

            list_size = len(list_A[idx][idx2])
            list_len.append(list_size)  # 각 cycle 마다의 데이터 길이값이 추가된다.
            #hist = hist+ np.ones(num_bin-1)  # prev 16



            tmp_hist.append(hist)
            tmp_bin_edge.append(bin_edges)

            #prob_list = np.array(hist) / ((list_size + (num_bin-1))/100)

            #prob_list = hist  #np.array(hist) / (list_size /100) # prev 16
            prob_list =  np.array(hist) / (list_size /100) # prev 16

            current_min = np.min(np.abs(current_list[idx][idx2])) - 0.3
            current_max = np.max(np.abs(current_list[idx][idx2])) + 0.3

            hist_cur, bin_edges_cur = np.histogram(np.abs(current_list[idx][idx2]), bins=np.linspace(current_min, current_max, 10))  # 16 bins    prev 17
            list_size_cur = len(current_list[idx][idx2])
            hist_cur = hist_cur + np.ones(9)

            prob_list_cur = hist_cur / (list_size_cur + 9)

            try:
                entropy_cycle = Entropy0(prob_list)
                entropy_cycle_cur = Entropy0(prob_list_cur)
            except:
                print("Exception")
                entropy_cycle = entropy_list[-1]
                entropy_cycle_cur = entropy_cycle_cur[-1]

            entropy_list.append(entropy_cycle)
            entropy_cur_list.append(entropy_cycle_cur)

        tmp_bin = []
        for bi in range(len(tmp_bin_edge)):
            tmp_bin.append([(tmp_bin_edge[bi][t]+tmp_bin_edge[bi][t+1])/2 for t in range(len(tmp_bin_edge[bi])-1)])

        '''
        Plot Voltage Distribution Histrogram
        '''
        # plt.figure()
        # plt.plot(tmp_bin[0], tmp_hist[0], label='cycle=0')
        # plt.plot(tmp_bin[100], tmp_hist[100], label='cycle=100')
        # plt.plot(tmp_bin[300], tmp_hist[300], label='cycle=300')
        # plt.plot(tmp_bin[500], tmp_hist[500], label='cycle=500')
        # plt.plot(tmp_bin[700], tmp_hist[700], label='cycle=700')
        # plt.legend()
        # plt.ylabel('Frequency')
        # plt.xlabel('Time (s)')
        # plt.show()

        check_curremt_sum = []
        for i in range(len(All_DC_time[idx])):
            if len(current_list[idx][i]) <= 1 or len(All_DC_time[idx][i]) == 0 :
                #print("list size is zero, idx : ", i)
                if i == 0:
                    check_curremt_sum.append(0)
                else:
                    check_curremt_sum.append(check_curremt_sum[-1])
                continue

            current = np.abs(current_list[idx][i][0])
            try:
                if current_list[idx][i][0] < -3.3:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i])+0.2, All_DC_time[idx][i])[-1]
                else:
                    integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i]), All_DC_time[idx][i])[-1]
                check_curremt_sum.append(integ_CB)
                #check_curremt_sum.append((((All_DC_time[idx][i][-1]) * current) + (223.618 * np.exp(0.977 * current)) - 259.77) / 100)
            except:
                print("All_DC_time[idx][i] :", All_DC_time[idx][i], " All_DC_time[idx][i][0] : ", All_DC_time[idx][i])
                #print("elapsed_time : ", elapsed_time, " current : ", current)

        # check_curremt_sum = Moving_Avg_Filter(check_curremt_sum[:], 5)
        # entropy_list = Moving_Avg_Filter(entropy_list[:], 5)
        # entropy_cur_list = Moving_Avg_Filter(entropy_cur_list[:], 5)

        ret_mul = []

        all_list.append(entropy_list)
        all_cap.append(check_curremt_sum)
        all_cur_list.append(entropy_cur_list)

    if mode is 'discharge':
        return all_list, all_cap, prob_list, list_A, curremt_sum_list, ret_mul, all_cur_list
    elif mode is 'charge':
        return all_list, retrun_prob_list


def EntropyForSOHProb_total_cur2(list_A, current_list, mode, SOH, All_DC_time):
    list_len = []
    retrun_prob_list = []

    all_list = []
    all_cap = []
    all_cur_list = []


    for idx in range(len(list_A)):  # cycle
        prob_list = []

        entropy_list = []
        curremt_sum_list = []

        entropy_cur_list = []

        for idx2 in range(len(list_A[idx])):  # cycle
            if len(current_list[idx][idx2]) == 0:
                continue
            if current_list[idx][idx2][0] > -2 or current_list[idx][idx2][0] < -2.4:
                continue

            if len(list_A[idx][idx2]) <= 1:
                if idx2 == 0:
                    entropy_list.append(0)
                else:
                    entropy_list.append(entropy_list[-1])
                continue

            # INR 18650 Battery Data
            #Ri =(-0.0357 * SOH[idx][idx2]) + 0.19  #0.15

            Ri = (-0.4722347 * SOH[idx][idx2]) + 0.62305638  # 62305638

            for i in range(len(list_A[idx][idx2])):
                try:
                    list_A[idx][idx2][i] = list_A[idx][idx2][i] + (np.multiply(Ri, np.abs(current_list[idx][idx2][i])))
                except:
                    print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx][idx2][i], "resistance : ", Ri)

            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(3.4, 4.2, 17))  # 16 bins    prev 17
            list_size = len(list_A[idx][idx2])
            list_len.append(list_size)  # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)  # prev 16

            prob_list = hist / (list_size + 16)  # prev 16

            current_min = np.min(np.abs(current_list[idx][idx2])) - 0.3
            current_max = np.max(np.abs(current_list[idx][idx2])) + 0.3


            hist_cur, bin_edges_cur = np.histogram(np.abs(current_list[idx][idx2]), bins=np.linspace(current_min, current_max, 10))  # 16 bins    prev 17
            list_size_cur = len(current_list[idx][idx2])
            hist_cur = hist_cur + np.ones(9)

            prob_list_cur = hist_cur / (list_size_cur + 9)

            try:
                entropy_cycle = Entropy0(prob_list)
                entropy_cycle_cur = Entropy0(prob_list_cur)
            except:
                entropy_cycle = entropy_list[-1]
                entropy_cycle_cur = entropy_cycle_cur[-1]

            entropy_list.append(entropy_cycle)
            entropy_cur_list.append(entropy_cycle_cur)

        check_curremt_sum = []
        for i in range(len(All_DC_time[idx])):
            if len(current_list[idx][i]) == 0:
                continue
            if current_list[idx][i][0] > -2 or current_list[idx][i][0] < -2.4:
                continue

            if len(current_list[idx][i]) <= 1 or len(All_DC_time[idx][i]) == 0 :
                #print("list size is zero, idx : ", i)
                if i == 0:
                    check_curremt_sum.append(0)
                else:
                    check_curremt_sum.append(check_curremt_sum[-1])
                continue

            current = np.abs(current_list[idx][i][0])
            try:
                integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i]), All_DC_time[idx][i])[-1]
                check_curremt_sum.append(integ_CB)
                #check_curremt_sum.append((((All_DC_time[idx][i][-1]) * current) + (223.618 * np.exp(0.977 * current)) - 259.77) / 100)
            except:
                print("All_DC_time[idx][i] :", All_DC_time[idx][i], " All_DC_time[idx][i][0] : ", All_DC_time[idx][i])
                #print("elapsed_time : ", elapsed_time, " current : ", current)

        check_curremt_sum = Moving_Avg_Filter(check_curremt_sum[:], 5)
        entropy_list = Moving_Avg_Filter(entropy_list[:], 5)
        #entropy_cur_list = Moving_Avg_Filter(entropy_cur_list[:], 5)

        ret_mul = []

        # for i in range(len(entropy_list)):
        #     entropy_list[i] = entropy_list[i] * 4300
        #     ret_mul.append((((entropy_list[i] ** (current)) * ((4200-check_curremt_sum[i]) ** (2.3-(current))))**(1/2.3))-(932.63*(0.11**(1.31))-1000))

        all_list.append(entropy_list)
        all_cap.append(check_curremt_sum)
        all_cur_list.append(entropy_cur_list)

    if mode is 'discharge':
        return all_list, all_cap, prob_list, list_A, curremt_sum_list, ret_mul, all_cur_list
    elif mode is 'charge':
        return all_list, retrun_prob_list


def EntropyForSOHProb_total_cur_charge(list_A, current_list, mode, SOH, All_DC_time):
    list_len = []
    retrun_prob_list = []

    all_list = []
    all_cap = []
    all_cur_list = []


    for idx in range(len(list_A)):  # cycle
        prob_list = []

        entropy_list = []
        curremt_sum_list = []

        entropy_cur_list = []

        for idx2 in range(len(list_A[idx])):  # cycle
            if len(list_A[idx][idx2]) <= 1:
                if idx2 == 0:
                    entropy_list.append(0)
                else:
                    entropy_list.append(entropy_list[-1])
                continue

            tmp_x = []

            for t in range(len(list_A[idx][1])):
                tmp_x.append(30 * t)

            # plt.figure()
            # plt.plot(tmp_x, list_A[idx][1])
            # plt.ylabel('Voltage (v)')
            # plt.xlabel('Time Stamp (s)')
            # plt.show()

            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(3.4, 4.18, 17))  # 16 bins    prev 17
            list_size = len(list_A[idx][idx2])
            list_len.append(list_size)  # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)  # prev 16

            prob_list = hist / (list_size + 16)  # prev 16

            current_min = np.min(np.abs(current_list[idx][idx2]))
            current_max = np.max(np.abs(current_list[idx][idx2]))


            hist_cur, bin_edges_cur = np.histogram(np.abs(current_list[idx][idx2]), bins=np.linspace(current_min, current_max, 17))  # 16 bins    prev 17
            list_size_cur = len(current_list[idx][idx2])
            hist_cur = hist_cur + np.ones(16)

            prob_list_cur = hist_cur / (list_size_cur + 16)

            try:
                entropy_cycle = Entropy0(prob_list)
                entropy_cycle_cur = Entropy0(prob_list_cur)
            except:
                entropy_cycle = entropy_list[-1]
                entropy_cycle_cur = entropy_cycle_cur[-1]

            entropy_list.append(entropy_cycle)
            entropy_cur_list.append(entropy_cycle_cur)

        check_curremt_sum = []
        for i in range(len(All_DC_time[idx])):
            if len(current_list[idx][i]) <= 1 or len(All_DC_time[idx][i]) == 0 :
                #print("list size is zero, idx : ", i)
                if i == 0:
                    check_curremt_sum.append(0)
                else:
                    check_curremt_sum.append(check_curremt_sum[-1])
                continue

            current = np.abs(current_list[idx][i][0])
            try:
                integ_CB = integrate.cumtrapz(np.abs(current_list[idx][i]), All_DC_time[idx][i])[-1]
                check_curremt_sum.append(integ_CB)
                #check_curremt_sum.append((((All_DC_time[idx][i][-1]) * current) + (223.618 * np.exp(0.977 * current)) - 259.77) / 100)
            except:
                print("All_DC_time[idx][i] :", All_DC_time[idx][i], " All_DC_time[idx][i][0] : ", All_DC_time[idx][i])
                #print("elapsed_time : ", elapsed_time, " current : ", current)

        check_curremt_sum = Moving_Avg_Filter(check_curremt_sum[:], 5)
        entropy_list = Moving_Avg_Filter(entropy_list[:], 5)
        #entropy_cur_list = Moving_Avg_Filter(entropy_cur_list[:], 5)

        ret_mul = []

        # for i in range(len(entropy_list)):
        #     entropy_list[i] = entropy_list[i] * 4300
        #     ret_mul.append((((entropy_list[i] ** (current)) * ((4200-check_curremt_sum[i]) ** (2.3-(current))))**(1/2.3))-(932.63*(0.11**(1.31))-1000))

        all_list.append(entropy_list)
        all_cap.append(check_curremt_sum)
        all_cur_list.append(entropy_cur_list)

    if mode is 'discharge':
        return all_list, all_cap, prob_list, list_A, curremt_sum_list, ret_mul, all_cur_list
    elif mode is 'charge':
        return all_list, retrun_prob_list


#######################################################################
'''Distribution Entropy구하는 방법'''
def UpperDM(U):
    N = len(U)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _UpperDM(m):
        x = [[U[j] for j in range(i, i+m-1+1)] for i in range(N-m+1)]
        D = [[_maxdist(x[i], x[j]) for j in range(N-m+1)] for i in range(N-m+1)]
        D = np.array(D)

        return D[np.triu_indices(N-m, 1)]

    UD = _UpperDM(10)

    return UD

def DistEN(M, list):
    all_entropy =[]
    for idx in range(len(list)):
        UDM = UpperDM(list[idx])
        hist, bin_edges = np.histogram(UDM, bins=np.linspace(0,2, M+1))
        hist = hist + np.ones(M)
        list_size = len(UDM) + M
        prob = hist/list_size
        all_entropy.append(-1*np.sum(prob*np.log2(prob))/np.log2(M))

    return all_entropy
##########################################################################
def concatenateEntropy(DCEnt, CEnt):
    '''
    :param DCEnt: DisCharge [# of train battery test, Entropy list]
    :param CEnt: charge [# of train battery test, Entropy list]
    :return ConcEntropy: [# of train battery test, ?, 2]

    '''
    ConcEntropy =[]
    for idx in range(len(DCEnt)):
        if len(DCEnt[idx]) == len(CEnt[idx]):
            DC_newaxis = np.array(DCEnt[idx])[:, np.newaxis]
            C_newaxis = np.array(CEnt[idx])[:, np.newaxis]
            ConcEntropy.append(list(np.concatenate((DC_newaxis,C_newaxis), axis=1)))
        else:
            print("Discharge Ent and Charge Ent length are different")
            break

    return ConcEntropy

def concatenateData3(data1, data2, data3):           # Entropy_withdischarge
    '''
    discharge ent와 charge ent를 이어줌.
    :param DCEnt: DisCharge [# of train battery test, [Entropy list, discharge_current]]
    :param CEnt: charge [# of train battery test, Entropy list]
    :return ConcEntropy: [# of train battery test, ?, 3]
    '''
    ConcEntropy =[]
    for idx in range(len(data1)):
        if len(data1[idx]) == len(data2[idx]):
            # DC_newaxis = np.array(data1[idx])[:, np.newaxis]
            try:
                np.shape(data1[idx])[1]
                data1_newaxis = data1[idx]
            except IndexError:
                data1_newaxis = np.array(data1[idx])[:, np.newaxis]
            try:
                np.shape(data2[idx])[1]
                data2_newaxis = data2[idx]
            except IndexError:
                data2_newaxis = np.array(data2[idx])[:, np.newaxis]
            try:
                np.shape(data3[idx])[1]
                data3_newaxis = data3[idx]
            except IndexError:
                data3_newaxis = np.array(data3[idx])[:, np.newaxis]

            ConcEntropy.append(list(np.concatenate((data1_newaxis, data2_newaxis, data3_newaxis), axis=1)))
        else:
            print("Discharge Ent and Charge Ent length are different")
            print("Ent len : ", len(data1[idx]), " Charge Ent : ", len(data2[idx]))
            break

    return ConcEntropy


def concatenateData4(data1, data2, data3, data4):           # concatenateEntropy_changedata
    '''
    :return ConcEntropy: [# of train battery test, ?, 4]
    '''
    ConcEntropy =[]
    for idx in range(len(data1)):
        if len(data1[idx]) == len(data2[idx]):
            # DC_newaxis = np.array(data1[idx])[:, np.newaxis]
            try:
                np.shape(data1[idx])[1]
                data1_newaxis = data1[idx]
            except IndexError:
                data1_newaxis = np.array(data1[idx])[:, np.newaxis]
            try:
                np.shape(data2[idx])[1]
                data2_newaxis = data2[idx]
            except IndexError:
                data2_newaxis = np.array(data2[idx])[:, np.newaxis]
            try:
                np.shape(data3[idx])[1]
                data3_newaxis = data3[idx]
            except IndexError:
                data3_newaxis = np.array(data3[idx])[:, np.newaxis]
            try:
                np.shape(data4[idx])[1]
                data4_newaxis = data4[idx]
            except IndexError:
                data4_newaxis = np.array(data4[idx])[:, np.newaxis]


            ConcEntropy.append(list(np.concatenate((data1_newaxis, data2_newaxis, data3_newaxis, data4_newaxis), axis=1)))
        else:
            print("Discharge Ent and Charge Ent length are different")
            print("Ent len : ", len(data1[idx]), " Charge Ent : ", len(data2[idx]))
            break

    return ConcEntropy

def concatenateData5(data1, data2, data3, data4, data5):           # concatenateEntropy_changedata
    '''
    :return ConcEntropy: [# of train battery test, ?, 4]
    '''
    ConcEntropy =[]
    for idx in range(len(data1)):
        if len(data1[idx]) == len(data2[idx]):
            # DC_newaxis = np.array(data1[idx])[:, np.newaxis]
            try:
                np.shape(data1[idx])[1]
                data1_newaxis = data1[idx]
            except IndexError:
                data1_newaxis = np.array(data1[idx])[:, np.newaxis]
            try:
                np.shape(data2[idx])[1]
                data2_newaxis = data2[idx]
            except IndexError:
                data2_newaxis = np.array(data2[idx])[:, np.newaxis]
            try:
                np.shape(data3[idx])[1]
                data3_newaxis = data3[idx]
            except IndexError:
                data3_newaxis = np.array(data3[idx])[:, np.newaxis]
            try:
                np.shape(data4[idx])[1]
                data4_newaxis = data4[idx]
            except IndexError:
                data4_newaxis = np.array(data4[idx])[:, np.newaxis]
            try:
                np.shape(data5[idx])[1]
                data5_newaxis = data4[idx]
            except IndexError:
                data5_newaxis = np.array(data5[idx])[:, np.newaxis]


            ConcEntropy.append(list(np.concatenate((data1_newaxis, data2_newaxis, data3_newaxis, data4_newaxis, data5_newaxis), axis=1)))
        else:
            print("Discharge Ent and Charge Ent length are different")
            print("Ent len : ", len(data1[idx]), " Charge Ent : ", len(data2[idx]))
            break

    return ConcEntropy

if __name__ == "__main__":
    """
    battery data의 entropy를 구해서 npy로 저장한다.
    """

    dir_path_CS2 = "../data/dis_current_constant/CS2_XX_0"
    dir_path_CX2 = "../data/dis_current_constant/CX2_XX_0"
    dir_path_K2 = "../data/dis_current_constant/K2_XX_0"
    dir_path_B0 = "../data/Nasa_data/BatteryAgingARC_change"

    dir_path = "../data/battery_device/Data_Excel_1224"


    #battery_list = ['B0005','B0006','B0018']

    battery_list = ['2_B_Battery']
    #battery_list = ['B0005','B0006','B0018','B0046', 'B0047','B0048','B0033']
    #battery_list = ['B0027']

    all_SOH_data = []
    all_CAP_data = []
    all_DC_data = []
    all_DC_current_data = []
    all_DC_time = []


    for battery in battery_list:
        battery_path = os.path.join(dir_path, battery)


        # if battery.__contains__('CS2_'):
        #     battery_path = os.path.join(dir_path_CS2, battery)
        # elif battery.__contains__('CX2_'):
        #     battery_path = os.path.join(dir_path_CX2, battery)
        # elif battery.__contains__('K2_'):
        #     battery_path = os.path.join(dir_path_K2, battery)
        # elif battery.__contains__('B0'):
        #     battery_path = os.path.join(dir_path_B0, battery)
        # else:
        #     continue

        DC_list = np.load(battery_path + '/discharge_data.npy', allow_pickle=True)
        DC_list_current = np.load(battery_path + '/discharge_current.npy', allow_pickle=True)
        C_list = np.load(battery_path + '/charge_data.npy', allow_pickle=True)
        C_list_current = np.load(battery_path + '/charge_current.npy', allow_pickle=True)
        capacity = np.load(battery_path + '/capacity.npy', allow_pickle=True)
        DC_time = np.load(battery_path + '/discharge_time_all.npy', allow_pickle=True)
        #DC_time = np.load(battery_path + '/discharge_time.npy', allow_pickle=True)
        #C_time = np.load(battery_path + '/charge_time.npy', allow_pickle=True)

        rated_cap = 3.25

        # if battery.__contains__('CS2_'):
        #     rated_cap = 1.1
        # elif battery.__contains__('CX2_'):
        #     rated_cap = 1.35
        # elif battery.__contains__('K2_'):
        #     rated_cap = 2.6
        # elif battery.__contains__('B0'):
        #     rated_cap = 2
        # else:
        #     rated_cap = 1.1


        SOH = capacity/rated_cap

        all_SOH_data.append(SOH)
        all_DC_data.append(DC_list)
        all_DC_current_data.append(DC_list_current)
        all_DC_time.append(DC_time)

    DC_Entropy, all_cap, DC_prob, list_A, curremt_sum_list, ret_mul, all_cur_list = EntropyForSOHProb_total_cur(all_DC_data, all_DC_current_data, 'discharge', all_SOH_data, all_DC_time)

        #DC_Entropy, DC_prob = EntropyForSOHProb_withCurrent_oneNasaBattery(DC_list, DC_list_current, 1500, 'discharge', DC_time, rated_cap, All_DC_time)


        #DC_Entropy, DC_prob = EntropyForSOHProb_withCurrent_oneNasaBattery_prev(DC_list, DC_list_current, 50, 'discharge', DC_time, rated_cap)

        #C_Entropy, C_prob = EntropyForSOHProb_withCurrent_oneNasaBattery_prev(C_list, C_list_current, 50, 'charge', C_time, rated_cap)

        #C_Entropy.append(0)

        #print("C_Entropy : ", np.shape(C_Entropy))

        # DC_Entropy, DC_prob = EntropyForSOHProb_withCurrent_oneBattery(DC_list, DC_list_current, 50, 'discharge')
        # C_Entropy, C_prob = EntropyForSOHProb_withCurrent_oneBattery(C_list, C_list_current, 50, 'charge')

        #ConcEntropy = concatenateEntropy_withdischarge2(DC_Entropy, C_Entropy)

        #print(np.shape(DC_prob))

    np.save(battery_path+'/discharge_Entropy.npy', DC_Entropy[0])
    np.save(battery_path + '/discharge_Charge.npy', all_cap[0])


        #np.save(battery_path + '/check_voltage.npy', list_A)

        # tmp_bins = np.linspace(2, 5, 17)
        # x_bins = np.zeros(16)

        # for i in range(16):
        #     x_bins[i] = (tmp_bins[i] + tmp_bins[i+1])/2
        #
        # plt.figure()
        # plt.bar(x_bins, DC_prob[2], width=0.15)
        # #plt.plot(C_Entropy)
        # plt.show()