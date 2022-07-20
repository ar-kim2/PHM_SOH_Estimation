import tensorflow as tf
import numpy as np
import os
import random

from tensorflow.contrib import rnn
from utils import BatteryDevice_dataloader as dataloader
from utils import Entropy
from SOH_Estimator.simple_lstm import _my_lstm_cell_origin

from utils.Entropy import Moving_Avg_Filter
import matplotlib.pyplot as plt

""" Battery의 방전 데이터를 통해 SOH 추정하는 모델 학습 """
class SOHestimator_discharge:
    def lstm_cell(self, num_units, name):
        cell = rnn.LSTMCell(num_units, state_is_tuple=True, activation=tf.tanh, name=name)
        return cell

    def weight_variable(self, shape, stddev=0.1, initial=None):
        if initial is None:
            initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(self, shape, init_bias=0.1, initial=None):
        if initial is None:
            initial = tf.constant(init_bias, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def __init__(self, data_load=True, feature_size=None, drop_out=None, test_case=0):
        if test_case == 0:
            self.test_case = 3
        else:
            self.test_case = test_case

        # Test case 1 : 1개의 Battery 60% dataTraining, 나머지 40% data Test
        # Test case 2 : 3개의 Battery 60% data Training, 나머지 60% data Test
        # Test case 3 : 2개의 Battery 전체 data Training, 1개의 Battery 전체 data Test
        if self.test_case == 1:
            self.train_battery_list = ['7_G_Battery']
        elif self.test_case == 2:
            self.train_battery_list = ['5_E_Battery', '6_F_Battery', '7_G_Battery']
        else:
            self.train_battery_list = ['5_E_Battery', '6_F_Battery']

        self.batch_size = 1024 #2048  # input 배치사이즈 [ Batch size, sequnece legnth, feature size(2) ]  # 256  # 512
        self.seq_length = 20
        self.feature_size = feature_size
        self.floor_num = 2 # LSTM 층 수
        self.value_dim = 1
        self.prediction_cycle = 1 # SOH를 하나만 추정하니까 1
        self.hidden_size = 5 # output size
        self.smooth_window_size = 20 # input을 smoothing 할 때 사용
        self.iteration_num = 2000 # 학습 횟수
        self.test_iteration = 500 # 학습할때 500번째 학습횟수마다 학습데이터에대한 실험결과를 보여주기 위함
        self.summary = [] # 학습과정 중 값 저장하는 list
        self.data_load = data_load # data load 여부. 일단은 True로 두고 사용
        self.lr = 0.002 # 학습율
        self.ent_mean = 0.7818
        self.ent_var = 0.0674
        self.charge_mean = 3253.161
        self.charge_var = 659.054
        self.current_mean = 0.892
        self.current_var = 0.266
        self.current_mean2 = 0.892
        self.current_var2 = 0.266
        self.current_mean3 = 0.892
        self.current_var3 = 0.266

        self.ent_min = 1.0093110500880342
        self.ent_max = 1.1588335440422561
        self.charge_min = 8279.25999
        self.charge_max = 11310.273599999995
        self.current_min = 1.3397248120300753
        self.current_max = 2.3746228346456695
        self.current_min2 = 0.827082796428829
        self.current_max2 = 0.9382355311436777
        self.current_min3 = 0.4216786884197127
        self.current_max3 = 1.083820848758048

        if self.test_case == 1 or self.test_case == 2:
            self.isRandom = True
        else:
            self.isRandom = False

        ### 저장 경로 ####
        self.log_path = "./Model"
        self.result_path = \
            os.path.join(self.log_path,"SOH_iter{}_seq{}_unit{}_f{}_predict{}_drop05_0523_discharge_input4_3"\
            .format(self.iteration_num, self.seq_length, self.hidden_size,
                    self.floor_num, self.prediction_cycle)) # 저장 폴더 이름
        if self.data_load is True:
            self.load_data()
        self.drop_out = drop_out # dropout rate
        if self.drop_out is True: # dropout 사용 여부
            self.train_keep_prob = 0.5 # 학습일 때는 0.5 확률로 dropout
        self.session = tf.Session()
        # model build(graph)
        self.build()
        tf.global_variables_initializer().run(session=self.session)
        self.saver = tf.train.Saver(self.var_list, max_to_keep=1)
        self.validate_saver = tf.train.Saver(max_to_keep=1)

    def data_loader(self, battery_list, isTraining = False):
        # data loader for discharging with constant current
        all_SOH_data = []
        all_CAP_data = []
        all_DC_data = []
        all_DC_current_data = []
        all_DC_time = []

        dir_path = "./data/battery_device/Data_Excel_1224"

        all_DC_current_avg = []
        all_DC_current_variance = []

        all_DC_current_ent = []

        smooth = True

        for battery in battery_list:  # 배터리 하나
            print('====Reading {} Battery Data for Train ===='.format(battery))

            BatteryDataDir = os.path.join(dir_path, battery)

            print("load START")
            capacity = np.load(BatteryDataDir + '/capacity.npy', allow_pickle=True)
            discharge_data = np.load(BatteryDataDir + '/discharge_data.npy', allow_pickle=True)
            DC_list_current = np.load(BatteryDataDir + '/discharge_current.npy', allow_pickle=True)
            DC_time = np.load(BatteryDataDir + '/discharge_time_all.npy', allow_pickle=True)

            degenerated_cycle = len(capacity)   # 노화된 Cycle??

            rated_cap = 3.25

            if smooth is True:
                # 데이터 앞 뒤에 값을 추가하여 Gaussian Filter shift 현상 방지
                append_capacity = np.append(np.full((self.smooth_window_size - 1, 1), capacity[0], dtype=np.float32),
                                            capacity)
                append_capacity = np.append(append_capacity,
                                            np.full((self.smooth_window_size, 1), capacity[-1], dtype=np.float32))
                all_CAP_data.append(
                    np.array(dataloader.smoothListGaussian(append_capacity, self.smooth_window_size))[:, np.newaxis])

                all_SOH_data.append(
                    np.array(dataloader.smoothListGaussian(append_capacity, self.smooth_window_size))[:, np.newaxis] / rated_cap)

            else:
                print("Are you sure using Raw SOH data?")
                c_i = np.array(range(0, len(capacity)))
                c_i = np.expand_dims(c_i, axis=0)
                capacity_i = np.concatenate(([capacity], c_i), 0)
                capacity_i = np.transpose(capacity_i, [1, 0])
                all_CAP_data.append(np.array(capacity)[:, np.newaxis])
                all_SOH_data.append(
                    np.array(capacity)[:,np.newaxis] / rated_cap)

            all_DC_data.append(discharge_data[:degenerated_cycle])  # [?(# of train_battery_list), ?(# of cycles in folder), ?(voltage list)]
            all_DC_current_data.append(DC_list_current[:degenerated_cycle])  # [?(# of train_battery_list), ?(# of cycles in folder), ?(voltage list)]
            all_DC_time.append(DC_time[:degenerated_cycle])

            DC_current_avg = []
            DC_current_variance = []

            for k in range(len(DC_list_current[:degenerated_cycle])):
                DC_current_avg.append(np.abs(np.average(DC_list_current[k])))
                DC_current_variance.append(np.std(DC_list_current[k]))

            all_DC_current_avg.append(DC_current_avg)
            all_DC_current_variance.append(DC_current_variance)


        # 모든 배터리 데이터 셋으 가져온 뒤에, Entropy Index를 계산

        DC_Entropy, DC_charge, DC_prob, _, _, DC_mul, DC_Cur_Entropy = Entropy.EntropyForSOHProb_Battery_Device_dvdt22(
            all_DC_data, all_DC_current_data, 'discharge', all_SOH_data, all_DC_time)

        # DC_Entropy, DC_charge, DC_prob, _, _, DC_mul, DC_Cur_Entropy = Entropy.EntropyForSOHProb_Battery_Device_dtdv(all_DC_data, all_DC_current_data, 'discharge', all_SOH_data, all_DC_time)

        for bi in range(len(battery_list)):  # 배터리 하나
            BatteryDataDir = os.path.join(dir_path, battery_list[bi])
            np.save(BatteryDataDir + '/discharge_Entropy_reverse.npy', DC_Entropy[bi])
            np.save(BatteryDataDir + '/discharge_Charge_2.npy', DC_charge[bi])

        #EntropyForSOHProb_total_cur

        # for jj in range(len(DC_Entropy)):
            # DC_Entropy[jj] = Moving_Avg_Filter(DC_Entropy[jj], 5)
            # DC_charge[jj] = Moving_Avg_Filter(DC_charge[jj], 5)
            # all_DC_current_avg[jj] = Moving_Avg_Filter(all_DC_current_avg[jj], 5)
            # all_DC_current_variance[jj] = Moving_Avg_Filter(all_DC_current_variance[jj], 5)

        if isTraining == True:
            # DC_Entropy shape : 4x800
            # DC_charge shape : 4x800
            DC_tmp = [element for array in DC_Entropy for element in array]

            self.ent_mean = np.average(DC_tmp)
            self.ent_var = np.std(DC_tmp)
            self.ent_min = np.min(DC_tmp)
            self.ent_max = np.max(DC_tmp)

            DC_tmp2 = [element for array in DC_charge for element in array]

            self.charge_mean = np.average(DC_tmp2)
            self.charge_var = np.std(DC_tmp2)
            self.charge_min = np.min(DC_tmp2)
            self.charge_max = np.max(DC_tmp2)

            DC_tmp3 = [element for array in all_DC_current_avg for element in array]

            self.current_mean = np.average(DC_tmp3)
            self.current_var = np.std(DC_tmp3)
            self.current_min = np.min(DC_tmp3)
            self.current_max = np.max(DC_tmp3)

            DC_tmp4 = [element for array in DC_Cur_Entropy for element in array]

            self.current_mean2 = np.average(DC_tmp4)
            self.current_var2 = np.std(DC_tmp4)
            self.current_min2 = np.min(DC_tmp4)
            self.current_max2 = np.max(DC_tmp4)

            DC_tmp5 = [element for array in all_DC_current_variance for element in array]

            self.current_mean3 = np.average(DC_tmp5)
            self.current_var3 = np.std(DC_tmp5)
            self.current_min3 = np.min(DC_tmp5)
            self.current_max3 = np.max(DC_tmp5)

        # Normalize DC_Entropy
        for i in range(len(DC_Entropy)):
            for j in range(len(DC_Entropy[i])):
                #DC_Entropy[i][j] = (DC_Entropy[i][j] - self.ent_mean)/self.ent_var
                DC_Entropy[i][j] = (DC_Entropy[i][j] - self.ent_min) / (self.ent_max-self.ent_min)

        # Normalize DC_charge
        for i in range(len(DC_charge)):
            for j in range(len(DC_charge[i])):
                #DC_charge[i][j] = (DC_charge[i][j] - self.charge_mean)/self.charge_var
                DC_charge[i][j] = (DC_charge[i][j] - self.charge_min)/(self.charge_max-self.charge_min)

        # Normalize DC_Current_Average
        for i in range(len(all_DC_current_avg)):
            for j in range(len(all_DC_current_avg[i])):
                #all_DC_current_avg[i][j] = (all_DC_current_avg[i][j] - self.current_mean)/self.current_var
                all_DC_current_avg[i][j] = (all_DC_current_avg[i][j] - self.current_min) / (self.current_max-self.current_min)

        # Normalize DC_Current_Standard_Variance / Entropy
        for i in range(len(DC_Cur_Entropy)):
            for j in range(len(DC_Cur_Entropy[i])):
                #DC_Cur_Entropy[i][j] = (DC_Cur_Entropy[i][j] - self.current_mean2)/self.current_var2
                DC_Cur_Entropy[i][j] = (DC_Cur_Entropy[i][j] - self.current_min2) / (self.current_max2-self.current_min2)

        # Normalize DC_Current_Standard_Variance / Entropy
        for i in range(len(all_DC_current_variance)):
            for j in range(len(all_DC_current_variance[i])):
                #DC_Cur_Entropy[i][j] = (DC_Cur_Entropy[i][j] - self.current_mean2)/self.current_var2
                all_DC_current_variance[i][j] = (all_DC_current_variance[i][j] - self.current_min3) / (self.current_max3-self.current_min3)

        C_Entropy = []

        for i in range(len(DC_Entropy)):
            C_tmp= np.zeros(len(DC_Entropy[i]))
            C_Entropy.append(C_tmp)

        return all_SOH_data, all_CAP_data, DC_Entropy, DC_charge, DC_prob, C_Entropy, all_DC_current_avg, all_DC_current_variance, DC_mul, DC_Cur_Entropy

    def load_data(self):
        # training, test data를 만드는 함수
        def train_data_make(result_list, input_list, future_len, seq_len, pred_cyc, test_sample):
            appended_list = np.append(np.full((seq_len - 1, np.shape(input_list)[-1]), input_list[0], dtype=np.float32),
                                      input_list, axis=0)

            if self.isRandom == True and len(test_sample) != 0:
                appended_list = [appended_list[idx:idx + seq_len + future_len] for idx in test_sample]
            else:
                appended_list = [appended_list[idx:idx + seq_len + future_len] for idx in
                                 range(len(appended_list) + 1 - seq_len - pred_cyc)]

            if result_list is None:
                result_list = appended_list
            else:
                result_list.extend(appended_list)

            return result_list, appended_list

        # for train
        SOH_train_data, CAP_train_data, DC_train_Entropy, DC_train_charge, DC_train_prob, C_train_entropy, DC_train_current_avg, DC_train_current_var, DC_train_mul, DC_train_current_ent = self.data_loader(
            self.train_battery_list, isTraining=True)
        # for validation
        # SOH_val_data, CAP_val_data, DC_val_Entropy, DC_val_charge, DC_val_prob, C_val_entropy, DC_test_current_avg, DC_test_current_var, DC_test_mul, DC_test_current_ent = self.data_loader(self.val_battery_list)

        # capacity and cycle index normalization
        self.norm_list = []

        # 입력으로 사용하는 Feature에 따라서 변경해주어야함.
        # self.InputEntropy = Entropy.concatenateData3(DC_train_Entropy, DC_train_charge, DC_train_current_avg)
        self.InputEntropy = Entropy.concatenateData4(DC_train_Entropy, DC_train_charge, DC_train_current_avg, DC_train_current_var)
        #self.InputEntropy =  Entropy.concatenateData5(DC_train_Entropy, DC_train_charge, DC_train_current_avg, DC_train_current_ent, DC_train_current_var)

        # self.ValInputEntropy = Entropy.concatenateData5(DC_val_Entropy, DC_val_charge, DC_test_current_avg, DC_test_current_ent, DC_test_current_var)
        # self.ValInputEntropy = Entropy.concatenateData3(DC_val_Entropy, DC_val_charge, DC_test_current_avg)

        self.InputEntProb = self.InputEntropy
        # self.ValInputEntProb = self.ValInputEntropy

        self.TrainP1_2 = None
        self.TrainLabelP2 = []

        # self.ValP1_2 = None
        # #self.ValP1_2 = []
        # self.ValLabelP2 = []

        self.TrainP1_1 = []
        self.TrainLabelP1 = []

        self.ValP1_1 = []
        self.ValLabelP1 = []

        dir_path_Battery = "./data/battery_device/Data_Excel_1224"

        for i, train_data in enumerate(self.InputEntProb):
            battery = self.train_battery_list[i]

            ## For Random
            BatteryDataDir = os.path.join(dir_path_Battery, battery)
            train_sample = np.load(BatteryDataDir + '/train_sample.npy', allow_pickle=True)

            # x_input
            self.TrainP1_2, _ = train_data_make(result_list=self.TrainP1_2,
                                                input_list=train_data,
                                                future_len=0,
                                                seq_len=self.seq_length,
                                                pred_cyc=0,
                                                test_sample=train_sample)

            if self.isRandom == True and len(train_sample) != 0 :
                SOH_train_data2 = np.array([SOH_train_data[i][idx+19] for idx in train_sample])
                self.TrainLabelP2.extend(SOH_train_data2[:, 0])
            else:
                self.TrainLabelP2.extend(SOH_train_data[i][:, 0])

        train_data_len = int(len(self.TrainP1_2) * 0.8)

        sample_list = list(range(len(self.TrainP1_2)))
        train_list = random.sample(sample_list, train_data_len)

        for i in range(len(self.TrainP1_2)):
            if train_list.__contains__(i):
                if self.TrainP1_1 == None:
                    self.TrainP1_1 = [np.array(self.TrainP1_2[i])]
                    self.TrainLabelP1 = [np.array(self.TrainLabelP2[i])]
                else:
                    self.TrainP1_1.append(np.array(self.TrainP1_2[i]))
                    self.TrainLabelP1.append(np.array(self.TrainLabelP2[i]))
            else:
                if self.ValP1_1 == None:
                    self.ValP1_1 = [np.array(self.TrainP1_2[i])]
                    self.ValLabelP1 = [np.array(self.TrainLabelP2[i])]
                else:
                    self.ValP1_1.append(self.TrainP1_2[i])
                    self.ValLabelP1.append(self.TrainLabelP2[i])

        self.TrainLabelP1 = np.array(self.TrainLabelP1)[:, np.newaxis]
        self.TrainLabelP1 = self.TrainLabelP1
        self.ValLabelP1 = np.array(self.ValLabelP1)[:, np.newaxis]
        self.ValLabelP1 = self.ValLabelP1

    def build(self):
        # grpah build
        self.out_list = []
        if self.feature_size is None:
            self.feature_size = np.shape(self.InputEntProb[0])[-1]
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.feature_size], name = 'x_input')
        self.Y = tf.placeholder(tf.float32, [None, 1], name="y_input")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.input_x = self.X
        self.out_list.append(self.input_x)
        self.feature_size = int(self.input_x.shape[2])
        #####################################################################################################
        # with tf.name_scope("SOH_Estimator") as SOH_scope:
        #     for idx in range(1, self.floor_num + 1):
        #         with tf.name_scope("LSTM_{}".format(idx)) as lstm_scope:
        #             self.cell = self.lstm_cell(self.hidden_size, name="cell_{}".format(idx))
        #             # self.inital_state = self.cell.zero_state(self.batch_size, tf.float32)
        #             self.lstm_output, _ = tf.nn.dynamic_rnn(self.cell, self.input_x, dtype=tf.float32)
        #
        #             self.out_list.append(self.lstm_output)
        #             self.input_x = self.lstm_output
        #
        #     with tf.name_scope("FC_layer") as FC_scope:
        #         fully_input = self.lstm_output[:, -1]
        #         Y_predict = tf.contrib.layers.fully_connected(fully_input, self.prediction_cycle,
        #                                                       activation_fn=tf.nn.leaky_relu,
        #                                                       scope=FC_scope)
        #         self.out_list.append(Y_predict)
        #         for var in tf.trainable_variables(FC_scope):
        #             fully_connected_summary = tf.summary.histogram(var.name, var)
        #             self.summary.append(fully_connected_summary)
        ######################################################################################################
        with tf.name_scope("SOH_Estimator2") as SOH_scope:
            for idx in range(1, self.floor_num + 1):
                with tf.name_scope("LSTM_{}".format(idx)) as lstm_scope:
                    self.cell = _my_lstm_cell_origin(self.feature_size, self.hidden_size, idx)
                    self.lstm_outputs, _states = self.cell(self.input_x)
                    self.out_list.append(self.lstm_outputs)
                    self.input_x = self.lstm_outputs
                    self.feature_size = self.lstm_outputs.shape[2]

            with tf.name_scope("Fully_Connected_1") as fully_connected_scope:
                if self.drop_out is True:
                    fully_input = self.lstm_outputs
                    fully_input = tf.nn.dropout(fully_input, keep_prob=self.keep_prob)
                    fully_input = tf.reshape(fully_input, [-1, self.seq_length*self.hidden_size])
                else:
                    fully_input = self.lstm_outputs[:, -1]
                self.Y_predict = tf.contrib.layers.fully_connected(fully_input, 1, activation_fn=tf.nn.leaky_relu,
                                                              scope=fully_connected_scope)
                self.out_list.append(self.Y_predict)
                for var in tf.trainable_variables(fully_connected_scope):
                    fully_connected_summary = tf.summary.histogram(var.name, var)
                    self.summary.append(fully_connected_summary)
        # with tf.name_scope("SOH_Estimator3") as SOH_scope:
        #     with tf.name_scope("Fully_Connected_1") as fully_connected_scope:
        #         fully_input = self.input_x   # self.lstm_outputs[:, -1]
        #         self.fc_outputs = tf.contrib.layers.fully_connected(fully_input, 1, activation_fn=tf.nn.leaky_relu,
        #                                                       scope=fully_connected_scope)
        #         self.out_list.append(self.fc_outputs)
        #         self.input_x = self.fc_outputs
        #         self.feature_size = self.fc_outputs.shape[2]
        #     for idx in range(1, self.floor_num + 1):
        #         with tf.name_scope("LSTM_{}".format(idx)) as lstm_scope:
        #             self.cell = _my_lstm_cell_origin(self.feature_size, self.hidden_size, idx)
        #             self.lstm_outputs, _states = self.cell(self.input_x)
        #             self.out_list.append(self.lstm_outputs)
        #
        #             self.input_x = self.lstm_outputs
        #             self.feature_size = self.lstm_outputs.shape[2]
        #     with tf.name_scope("Fully_Connected_2") as fully_connected_scope:
        #         if self.drop_out is True:
        #             fully_input = self.lstm_outputs
        #             fully_input = tf.nn.dropout(fully_input, keep_prob=self.keep_prob)
        #             fully_input = tf.reshape(fully_input, [-1, self.seq_length*self.hidden_size])
        #         else:
        #             fully_input = self.lstm_outputs[:, -1]
        #         self.Y_predict = tf.contrib.layers.fully_connected(fully_input, 1, activation_fn=tf.nn.leaky_relu,
        #                                                       scope=fully_connected_scope)
        #         self.out_list.append(self.Y_predict)
        #         for var in tf.trainable_variables(fully_connected_scope):
        #             fully_connected_summary = tf.summary.histogram(var.name, var)
        #             self.summary.append(fully_connected_summary)
        ####################################################################################################
        self.var_list = tf.trainable_variables(scope=None) # 그래프 내의 변수 리스트를 불러온다.
        self.var_list_weight = list(set(self.var_list)-set(self.norm_list))
        with tf.name_scope("Optimizer"):
            # l2 normalization
            self.l2 = 0.00001 * sum(tf.nn.l2_loss(tf_var) for tf_var in self.var_list_weight)
            self.cost = tf.reduce_mean(tf.square(self.out_list[-1] - self.Y)) / 2 + self.l2
            self.cost_scalar = tf.summary.scalar(name='cost', tensor=self.cost)

            self.residual = tf.reduce_sum(tf.square(tf.subtract(self.Y, self.out_list[-1])))
            self.total = tf.reduce_sum(tf.square(tf.subtract(self.Y, tf.reduce_mean(self.Y))))
            self.rsquare = tf.subtract(1.0, tf.div(self.residual, self.total))
            self.rsquare_scalar = tf.summary.scalar(name='RSQUARE', tensor=self.rsquare)

            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.out_list[-1] - self.Y)))
            self.rmse_scalar = tf.summary.scalar(name='RMSE', tensor=self.rmse)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.training = self.optimizer.minimize(self.cost, var_list=None)
            #self.training = self.optimizer.minimize(-self.rsquare, var_list=None)
            #self.training = self.optimizer.minimize(self.rmse, var_list=None)

        self.merged_summary = tf.summary.merge(self.summary)
        self.rmse_summary = tf.summary.merge([self.rmse_scalar])
        self.cost_summary = tf.summary.merge([self.cost_scalar])

    def train(self):
        self.saver_path = self.result_path + "/saver"
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        else:
            print("Project already exists : {}".format(self.result_path))
            exit()
        if not os.path.exists(self.saver_path):
            os.makedirs(self.saver_path)
            print("Save Folder_1 Created!")

        self.best_cost = 99999.0
        self.best_validate = 99999.0
        self.number_of_batch = int(len(self.TrainP1_1) / self.batch_size) + 1
        print("Number of Batch : {}".format(self.number_of_batch))
        print("#########################Training START#########################")
        writer = tf.summary.FileWriter(self.result_path + "/train_phase" + "/train", self.session.graph)
        val_writer = tf.summary.FileWriter(self.result_path + "/train_phase" + "/validation", self.session.graph)

        Trigger = 0

        for iteration in range(1, self.iteration_num + 1):
            train_predict = []
            self.current_total_cost = 0.0
            # train
            for current_batch in range(0, len(self.TrainP1_1), self.batch_size):

                s, h, current_cost, _ = \
                    self.session.run([self.merged_summary, self.out_list, self.cost, self.training],
                                    feed_dict = {self.X: self.TrainP1_1[current_batch:current_batch+self.batch_size],
                                                 self.Y: self.TrainLabelP1[current_batch:current_batch+self.batch_size],
                                                 self.keep_prob: self.train_keep_prob})
                writer.add_summary(s, global_step=iteration)
                writer.flush()
                self.current_total_cost += current_cost/self.number_of_batch
                train_predict.extend(h[-1])

            if iteration % 10 == 0:
                print("step:", iteration, "COST: {}".format(self.current_total_cost))

            cost_summary_result = self.session.run(self.cost_summary, feed_dict={self.cost: self.current_total_cost})
            writer.add_summary(cost_summary_result, global_step=iteration)
            writer.flush()

            # y_validate_result = self.session.run(self.out_list[-1], feed_dict={self.X:self.ValP1_1})
            # validation test
            v_s, r_s, v_c, v_cs = self.session.run([self.rmse_summary, self.rmse, self.cost, self.cost_summary],
                                             feed_dict={self.X:self.ValP1_1, self.Y:self.ValLabelP1,
                                                        self.keep_prob:1})


            # v_s, r_s, v_c, v_cs = self.session.run([self.rmse_summary, self.rsquare, self.cost, self.cost_summary],
            #                                      feed_dict={self.X:self.ValP1_1, self.Y:self.ValLabelP1,
            #                                                 self.keep_prob:1})


            # writer.add_summary(v_cs, global_step=iteration)
            val_writer.add_summary(v_cs, global_step=iteration)
            val_writer.add_summary(v_s, global_step=iteration)
            val_writer.flush()


            if self.best_validate > r_s: # validation 결과가 제일 좋을때마다 모델을 저장
                print("Best Validate Step :", iteration, "RMSE : {} -> {}".format(self.best_validate, r_s))
                self.best_validate = r_s
                self.validate_saver.save(self.session, self.saver_path + "/bestvalidate/bestvalidate", global_step=iteration)
                Trigger = 0

            if self.best_cost > self.current_total_cost: # 학습 cost가 제일 낮을때마다 모델을 저장
                self.saver.save(self.session, self.saver_path + "/best/bestcost", global_step=iteration)
                print("Best Cost Step :", iteration,
                      "{} -> {}, RMSE : {}".format(self.best_cost, self.current_total_cost, r_s))
                self.best_cost = self.current_total_cost
                Trigger += 1

            if iteration % self.test_iteration == 0: # 500번째마다 결과 표시, 모델 저장
                test_saver = tf.train.Saver(max_to_keep=1)
                print("Save Test Step:", iteration, "{}".format(self.current_total_cost))
                test_saver.save(self.session, self.saver_path + "/test_{:0>5}/test".format(iteration), global_step=iteration)

            if Trigger > 4000: # overfitting 발생 시 멈춤
                writer.close()
                val_writer.close()
                print("OVERFITTING! Step : {}".format(iteration))
                break

    def load_model(self, model):
        # Example: self.load_model('AE_model/intermediate/fine_tune_epoch_1400.cptk')
        # model = os.path.join(model, 'saver/bestvalidate')
        model = os.path.join(model, 'saver/best')
        #model = os.path.join(model, 'saver/test_08000')

        self.saver.restore(self.session, tf.train.latest_checkpoint(model))

if __name__ == "__main__":
    Estimator = SOHestimator_discharge(feature_size=4, data_load=True, drop_out=True)
    Estimator.train()
