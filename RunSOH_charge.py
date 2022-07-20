from SOHestimatior_charge import SOHestimator_charge
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import Entropy

testcase = 3

Model_Dir = os.path.abspath('./Model')
Model_Dir = os.path.join(Model_Dir, 'SOH_iter1000_seq20_unit5_f2_predict1_drop03_0524_charge_input4_{}'.format(testcase))
Estimator = SOHestimator_charge(data_load=True, feature_size=4, drop_out=True, test_case=testcase) # SOH 추정 모델 클래스 로드
Estimator.load_model(model=Model_Dir) # 학습된 추정 모델 로드

if testcase == 1 or testcase == 2:
    isSampleData = True
else:
    isSampleData = False

selected_cycle = 1    # Estimator가 해당 cycle의 SOH추정
#####################################################################################
test_battery_list = ['7_G_Battery']

SOH_data, CAP_data, DC_Entropy, DC_charge, DC_prob, C_entropy, DC_current_avg, DC_current_var, DC_mul, DC_current_ent = Estimator.data_loader(test_battery_list)

# for i in range(len(DC_Entropy)):
#     plt.figure()
#     plt.plot(DC_Entropy[i])
#     plt.plot(DC_Entropy[i])
#
#     plt.title('Entropy {} (CS2_3)'.format(i))
#     plt.show()
#
# for i in range(len(DC_charge)):
#     plt.figure()
#     plt.plot(DC_charge[i])

#     plt.title('charge {} (CS2_3)'.format(i))
#     plt.show()



# DC_Entropy : [Battery_idx][Entropy]

# 방전전류가변화하는 데이터셋
# test_battery_list = ['CS2_3']
# CAP_data, DC_Entropy, DC_prob, C_entropy = Estimator.data_loader_changing(test_battery_list)
############################################################################################
# 엔트로피 인덱스로부터 input 생성
# InputEntropy = []
#
# for battery in test_battery_list:
#     InputEntropy.append(np.load(dir_path+battery+"/concEntropy.npy", allow_pickle=True))

# InputEntropy = Entropy.concatenateData3(DC_Entropy, DC_charge, DC_current_avg) # DC_Entropy #
InputEntropy = Entropy.concatenateData4(DC_Entropy, DC_charge, DC_current_avg, DC_current_var)
#############################################################################################
# 실제 capacity를 구하기 위해 이렇게 한다함...
# Entmin_value= Estimator.session.run("Ent_min_value:0")
# Entmax_value= Estimator.session.run("Ent_max_value:0")
# InputEntropy = tools.MinMaxScaler(InputEntropy, Entmin_value, Entmax_value)
InputEntProb = InputEntropy

def test_data_make(result_list, input_list, future_len, seq_len, pred_cyc):
    # Input [?, seq_length, feature_size] 형태로 변환
    appended_list = np.append(np.full((seq_len - 1, np.shape(input_list)[-1]), input_list[0], dtype=np.float32),input_list, axis=0)

    test_sample = np.load("./data/battery_device/Data_Excel_1224/7_G_Battery/test_sample.npy", allow_pickle=True)
    #test_sample = np.load("./data/battery_device/Data_Excel_0910/7_G_Battery/test_sample.npy", allow_pickle=True)

    if isSampleData == True:
        appended_list = [appended_list[idx:idx + seq_len + future_len] for idx in test_sample]
    else:
        appended_list = [appended_list[idx:idx + seq_len + future_len] for idx in range(len(appended_list) + 1 - seq_len - pred_cyc)]

    if result_list is None:
        result_list = appended_list
    else:
        result_list.extend(appended_list)

    return result_list, appended_list

TestP1_1 = None
TestLabelP1 = []

for i, _data in enumerate(InputEntProb):
    TestP1_1, _ = test_data_make(result_list=TestP1_1,
                                      input_list=_data,
                                      future_len=0,
                                      seq_len=Estimator.seq_length,
                                      pred_cyc=0)

    test_sample = np.load("./data/battery_device/Data_Excel_1224/7_G_Battery/test_sample.npy", allow_pickle=True)
    SOH_data2 = np.array([SOH_data[0][idx + 19] for idx in test_sample])

    if isSampleData == True:
        TestLabelP1.extend(SOH_data2[:, 0])
    else:
        TestLabelP1.extend(SOH_data[i][:, 0])


print(np.shape(TestLabelP1))
print(TestLabelP1[0])

for i in range(len(TestLabelP1)):
    TestLabelP1[i] = TestLabelP1[i]

TestP1_1 = TestP1_1[:]
TestLabelP1 = TestLabelP1[:]

############################### 전체 cycle에서 SOH추정 결과 #################################
Estimated_SOH = Estimator.session.run(Estimator.out_list[-1],
                      feed_dict={Estimator.X : TestP1_1,
                                 Estimator.keep_prob: 1})

print("Estimated_SOH shape : ", np.shape(Estimated_SOH))
# print(Estimated_SOH)

ln1 = plt.plot(TestLabelP1[:], label='real capacity')
ln2 = plt.plot(Estimated_SOH[:], label ='estimated cpacity')
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc='upper right',fontsize=10)
plt.xlabel('Cycles', fontsize=14)
plt.ylabel('SOH(%)', fontsize=14)
# plt.ylim([0.2, 1.2])
plt.show()
# print("estimate2 :", np.ravel(Estimated_SOH))
#np.save("./data/dis_current_constant/" + '/cs2_8_rul2', np.ravel(Estimated_SOH))


#########################################################################################3
print("rmse : ", Estimator.session.run(Estimator.rmse, feed_dict={Estimator.X : TestP1_1,
                                                       Estimator.Y: np.array(TestLabelP1)[:, np.newaxis],
                                                       Estimator.keep_prob:1}))

print("r square : ", Estimator.session.run(Estimator.rsquare, feed_dict={Estimator.X : TestP1_1,
                                                       Estimator.Y: np.array(TestLabelP1)[:, np.newaxis],
                                                       Estimator.keep_prob:1}))

Estimated_SOH = Estimator.session.run(Estimator.out_list[-1],
                      feed_dict={Estimator.X : np.array(TestP1_1[selected_cycle])[np.newaxis, :, :],
                                 Estimator.keep_prob:1})
print("estimate :", Estimated_SOH[0,0])
# calculate SOH at cycle k => SOH_k : 구현완료


########################################################################################

