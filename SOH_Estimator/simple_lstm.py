import tensorflow as tf
import numpy as np
class _my_lstm_cell_origin():
    def __init__(self, feature_size, hidden_size, idx):
        self._hidden_size = hidden_size
        self._feature_size = int(feature_size)

        with tf.variable_scope("lstm_cell_no_{}".format(idx)):
            '''input gate weights
            f : feature_size
            h : hidden_size
            Wi = ( f + h , h )
            '''
            self.Wxi = tf.get_variable('Wxi', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whi = tf.get_variable('Whi', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wci = tf.get_variable('Wci', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wi = tf.concat([self.Wxi, self.Whi], axis=0)

            '''forget gate weights
            Wf = ( f + h  , h )
            '''
            self.Wxf = tf.get_variable('Wxf', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whf = tf.get_variable('Whf', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wcf = tf.get_variable('Wcf', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wf = tf.concat([self.Wxf, self.Whf], axis=0)

            '''cell update weights
            Wc = ( f + h , h )
            '''
            self.Wxc = tf.get_variable('Wxc', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whc = tf.get_variable('Whc', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wc = tf.concat([self.Wxc, self.Whc], axis=0)

            '''output gate weights
            Wc = ( f + h  , h )
            '''
            self.Wxo = tf.get_variable('Wxo', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Who = tf.get_variable('Who', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wco = tf.get_variable('Wco', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wo = tf.concat([self.Wxo, self.Who], axis=0)

            '''bias term for all gates'''
            self.bi = tf.get_variable('bi', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bf = tf.get_variable('bf', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bc = tf.get_variable('bc', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bo = tf.get_variable('bo', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    def in_fo_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: gate output
        '''
        ig_result = tf.sigmoid(tf.matmul(x, self.Wi) + self.bi)
        fg_result = tf.sigmoid(tf.matmul(x, self.Wf) + self.bf)

        return ig_result, fg_result

    def cell_update(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: cell state updating term
        '''
        return tf.tanh(tf.matmul(x, self.Wc) + self.bc)

    def ou_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t)
        :return: output gate output
        '''
        return tf.sigmoid(tf.matmul(x, self.Wo) + self.bo)

    def step(self, prev, input_t):
        h_state, c_state = tf.unstack(prev)
        concat_x = tf.concat([input_t, h_state], axis=1)
        ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
        cu = _my_lstm_cell.cell_update(self, concat_x)

        cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
        og = _my_lstm_cell.ou_gate(self, concat_x)

        h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
        return tf.stack([h_state_new, cell_state_new])

    def __call__(self, rnn_input): # no keep_prob_list
        # rnn input : [?, seq_length, hidden_size]
        zeros_dims = tf.stack([2, tf.shape(rnn_input)[0], self._hidden_size])
        init_state = tf.fill(zeros_dims, 0.0)
        self._seq_length = rnn_input.shape[1]

        # scan은 반드시 init_state의 dimension 과 fn(step)의 리턴값의 dimension 이 동일해야한다.
        # scan의 initializer term은 step의 첫번째 인수에 해당한다.
        # transpose to (seq_length, ?, feature_size)
        h_and_c = tf.scan(self.step, tf.transpose(rnn_input, [1, 0, 2]), initializer=init_state)

        # need to transpose (seq_length, 2, ?, feature_size) -> (2, ?, seq_length, feature_size)
        hidden_states, cell_states = tf.unstack(tf.transpose(h_and_c, [1, 2, 0, 3]))

        return hidden_states, cell_states

class _my_lstm_cell_fix_mask2():
    '''미리 복사된 마스크를 받아서'''
    def __init__(self, feature_size, hidden_size, idx, idx2 = 1):
        self._hidden_size = hidden_size
        self._feature_size = int(feature_size)

        with tf.variable_scope("lstm_cell_no_{}_{}".format(idx, idx2)):
            '''input gate weights
            f : feature_size
            h : hidden_size
            Wi = ( f + h , h )
            '''
            self.Wxi = tf.get_variable('Wxi', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whi = tf.get_variable('Whi', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wci = tf.get_variable('Wci', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wi = tf.concat([self.Wxi, self.Whi], axis=0)

            '''forget gate weights
            Wf = ( f + h  , h )
            '''
            self.Wxf = tf.get_variable('Wxf', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whf = tf.get_variable('Whf', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wcf = tf.get_variable('Wcf', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wf = tf.concat([self.Wxf, self.Whf], axis=0)

            '''cell update weights
            Wc = ( f + h , h )
            '''
            self.Wxc = tf.get_variable('Wxc', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whc = tf.get_variable('Whc', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wc = tf.concat([self.Wxc, self.Whc], axis=0)

            '''output gate weights
            Wc = ( f + h  , h )
            '''
            self.Wxo = tf.get_variable('Wxo', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Who = tf.get_variable('Who', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wco = tf.get_variable('Wco', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wo = tf.concat([self.Wxo, self.Who], axis=0)

            '''bias term for all gates'''
            self.bi = tf.get_variable('bi', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bf = tf.get_variable('bf', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bc = tf.get_variable('bc', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bo = tf.get_variable('bo', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    def in_fo_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: gate output
        '''
        ig_result = tf.sigmoid(tf.matmul(x, self.Wi) + self.bi)
        fg_result = tf.sigmoid(tf.matmul(x, self.Wf) + self.bf)

        return ig_result, fg_result

    def cell_update(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: cell state updating term
        '''
        return tf.tanh(tf.matmul(x, self.Wc) + self.bc)

    def ou_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t)
        :return: output gate output
        '''
        return tf.sigmoid(tf.matmul(x, self.Wo) + self.bo)

    # def step(self, prev, input_t):
    #     h_state, c_state = tf.unstack(prev)
    #     concat_x = tf.concat([input_t, h_state], axis=1)
    #     ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
    #     cu = _my_lstm_cell.cell_update(self, concat_x)
    #
    #     cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
    #     og = _my_lstm_cell.ou_gate(self, concat_x)
    #
    #     h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
    #     return tf.stack([h_state_new, cell_state_new])
    def step(self, prev, input_and_mask):
        # concat의 반대는 split
        # 'value' is a tensor with shape [5, 30]
        # split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
        input_t, mask = tf.split(input_and_mask, [self._feature_size, self._hidden_size], axis=1)
        h_state, c_state = tf.unstack(prev)
        concat_x = tf.concat([input_t, h_state], axis=1)

        ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
        cu = _my_lstm_cell.cell_update(self, concat_x)

        cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
        cell_state_new = tf.multiply(cell_state_new, mask)
        og = _my_lstm_cell.ou_gate(self, concat_x)

        h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
        return tf.stack([h_state_new, cell_state_new])

    def __call__(self, rnn_input, mask_list):
        #rnn input : [?, seq_length, feature_size]
        #keep_prob_list : [?, seq_length, hidden_size] : mask
        self._seq_length = rnn_input.shape[1]
        self._NONE = tf.shape(rnn_input)[0]
        # keep_prob_list = tf.transpose(keep_prob_list, [1, 0, 2]) # [seq_length, ?, 1]
        # keep_prob_list = tf.reshape(keep_prob_list, [self._seq_length, -1])

        #mask generation
        masks = tf.transpose(mask_list, [1, 0, 2]) # [seq_length, ?, hidden_size)

        #init state
        zeros_dims = tf.stack([2, tf.shape(rnn_input)[0], self._hidden_size])
        init_state = tf.fill(zeros_dims, 0.0)

        rnn_input = tf.transpose(rnn_input, [1, 0, 2])
        concat_input = tf.concat([rnn_input, masks], axis=2) # [ seq_length, ?, f + h]

        # scan은 반드시 init_state의 dimension 과 fn(step)의 리턴값의 dimension 이 동일해야한다.
        # scan의 initializer term은 step의 첫번째 인수에 해당한다.
        # transpose to (seq_length, ?, feature_size)
        h_and_c = tf.scan(self.step, concat_input, initializer=init_state)

        # need to transpose (seq_length, 2, ?, feature_size) -> (2, ?, seq_length, feature_size)
        hidden_states, cell_states = tf.unstack(tf.transpose(h_and_c, [1, 2, 0, 3]))

        return hidden_states, cell_states

class _my_lstm_cell_fix_mask():
    def __init__(self, feature_size, hidden_size, idx, idx2 = 1):
        self._hidden_size = hidden_size
        self._feature_size = int(feature_size)

        with tf.variable_scope("lstm_cell_no_{}_{}".format(idx, idx2)):
            '''input gate weights
            f : feature_size
            h : hidden_size
            Wi = ( f + h , h )
            '''
            self.Wxi = tf.get_variable('Wxi', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whi = tf.get_variable('Whi', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wci = tf.get_variable('Wci', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wi = tf.concat([self.Wxi, self.Whi], axis=0)

            '''forget gate weights
            Wf = ( f + h  , h )
            '''
            self.Wxf = tf.get_variable('Wxf', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whf = tf.get_variable('Whf', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wcf = tf.get_variable('Wcf', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wf = tf.concat([self.Wxf, self.Whf], axis=0)

            '''cell update weights
            Wc = ( f + h , h )
            '''
            self.Wxc = tf.get_variable('Wxc', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whc = tf.get_variable('Whc', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wc = tf.concat([self.Wxc, self.Whc], axis=0)

            '''output gate weights
            Wc = ( f + h  , h )
            '''
            self.Wxo = tf.get_variable('Wxo', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Who = tf.get_variable('Who', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wco = tf.get_variable('Wco', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wo = tf.concat([self.Wxo, self.Who], axis=0)

            '''bias term for all gates'''
            self.bi = tf.get_variable('bi', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bf = tf.get_variable('bf', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bc = tf.get_variable('bc', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bo = tf.get_variable('bo', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    def in_fo_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: gate output
        '''
        ig_result = tf.sigmoid(tf.matmul(x, self.Wi) + self.bi)
        fg_result = tf.sigmoid(tf.matmul(x, self.Wf) + self.bf)

        return ig_result, fg_result

    def cell_update(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: cell state updating term
        '''
        return tf.tanh(tf.matmul(x, self.Wc) + self.bc)

    def ou_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t)
        :return: output gate output
        '''
        return tf.sigmoid(tf.matmul(x, self.Wo) + self.bo)

    # def step(self, prev, input_t):
    #     h_state, c_state = tf.unstack(prev)
    #     concat_x = tf.concat([input_t, h_state], axis=1)
    #     ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
    #     cu = _my_lstm_cell.cell_update(self, concat_x)
    #
    #     cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
    #     og = _my_lstm_cell.ou_gate(self, concat_x)
    #
    #     h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
    #     return tf.stack([h_state_new, cell_state_new])
    def step(self, prev, input_and_mask):
        # concat의 반대는 split
        # 'value' is a tensor with shape [5, 30]
        # split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
        input_t, mask = tf.split(input_and_mask, [self._feature_size, self._hidden_size], axis=1)
        h_state, c_state = tf.unstack(prev)
        concat_x = tf.concat([input_t, h_state], axis=1)

        ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
        cu = _my_lstm_cell.cell_update(self, concat_x)

        cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
        cell_state_new = tf.multiply(cell_state_new, mask)
        og = _my_lstm_cell.ou_gate(self, concat_x)

        h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
        return tf.stack([h_state_new, cell_state_new])

    def __call__(self, rnn_input, keep_prob_list):
        #rnn input : [?, seq_length, feature_size]
        #keep_prob_list : [?, hidden_size] : mask
        self._seq_length = rnn_input.shape[1]
        self._NONE = tf.shape(rnn_input)[0]
        # keep_prob_list = tf.transpose(keep_prob_list, [1, 0, 2]) # [seq_length, ?, 1]
        # keep_prob_list = tf.reshape(keep_prob_list, [self._seq_length, -1])
        #mask generation

        masks = tf.cast(keep_prob_list, tf.float32)
        masks = tf.reshape(masks, [-1, 1, self._hidden_size])
        masks_list = tf.map_fn(lambda x: tf.tile(x, [100, 1]), masks)

        masks_list = tf.transpose(masks_list, [1,0,2])

        zeros_dims = tf.stack([2, tf.shape(rnn_input)[0], self._hidden_size])
        init_state = tf.fill(zeros_dims, 0.0)

        rnn_input = tf.transpose(rnn_input, [1, 0, 2])
        concat_input = tf.concat([rnn_input, masks_list], axis=2) # [ seq_length, ?, f + h]

        # scan은 반드시 init_state의 dimension 과 fn(step)의 리턴값의 dimension 이 동일해야한다.
        # scan의 initializer term은 step의 첫번째 인수에 해당한다.
        # transpose to (seq_length, ?, feature_size)
        h_and_c = tf.scan(self.step, concat_input, initializer=init_state)

        # need to transpose (seq_length, 2, ?, feature_size) -> (2, ?, seq_length, feature_size)
        hidden_states, cell_states = tf.unstack(tf.transpose(h_and_c, [1, 2, 0, 3]))

        return hidden_states, cell_states

class _my_lstm_cell():
    def __init__(self, feature_size, hidden_size, idx):
        self._hidden_size = hidden_size
        self._feature_size = int(feature_size)

        with tf.variable_scope("lstm_cell_no_{}".format(idx)):
            '''input gate weights
            f : feature_size
            h : hidden_size
            Wi = ( f + h , h )
            '''
            self.Wxi = tf.get_variable('Wxi', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whi = tf.get_variable('Whi', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wci = tf.get_variable('Wci', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wi = tf.concat([self.Wxi, self.Whi], axis=0)

            '''forget gate weights
            Wf = ( f + h  , h )
            '''
            self.Wxf = tf.get_variable('Wxf', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whf = tf.get_variable('Whf', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wcf = tf.get_variable('Wcf', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wf = tf.concat([self.Wxf, self.Whf], axis=0)

            '''cell update weights
            Wc = ( f + h , h )
            '''
            self.Wxc = tf.get_variable('Wxc', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whc = tf.get_variable('Whc', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wc = tf.concat([self.Wxc, self.Whc], axis=0)

            '''output gate weights
            Wc = ( f + h  , h )
            '''
            self.Wxo = tf.get_variable('Wxo', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Who = tf.get_variable('Who', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wco = tf.get_variable('Wco', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wo = tf.concat([self.Wxo, self.Who], axis=0)

            '''bias term for all gates'''
            self.bi = tf.get_variable('bi', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bf = tf.get_variable('bf', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bc = tf.get_variable('bc', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bo = tf.get_variable('bo', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    def in_fo_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: gate output
        '''
        ig_result = tf.sigmoid(tf.matmul(x, self.Wi) + self.bi)
        fg_result = tf.sigmoid(tf.matmul(x, self.Wf) + self.bf)

        return ig_result, fg_result

    def cell_update(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: cell state updating term
        '''
        return tf.tanh(tf.matmul(x, self.Wc) + self.bc)

    def ou_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t)
        :return: output gate output
        '''
        return tf.sigmoid(tf.matmul(x, self.Wo) + self.bo)

    # def step(self, prev, input_t):
    #     h_state, c_state = tf.unstack(prev)
    #     concat_x = tf.concat([input_t, h_state], axis=1)
    #     ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
    #     cu = _my_lstm_cell.cell_update(self, concat_x)
    #
    #     cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
    #     og = _my_lstm_cell.ou_gate(self, concat_x)
    #
    #     h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
    #     return tf.stack([h_state_new, cell_state_new])
    def step(self, prev, input_and_mask):
        # concat의 반대는 split
        # 'value' is a tensor with shape [5, 30]
        # split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
        input_t, mask = tf.split(input_and_mask, [self._feature_size, self._hidden_size], axis=1)
        h_state, c_state = tf.unstack(prev)
        concat_x = tf.concat([input_t, h_state], axis=1)

        ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
        cu = _my_lstm_cell.cell_update(self, concat_x)

        cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
        cell_state_new = tf.multiply(cell_state_new, mask)
        og = _my_lstm_cell.ou_gate(self, concat_x)

        h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
        return tf.stack([h_state_new, cell_state_new])

    def __call__(self, rnn_input, keep_prob_list):
        #rnn input : [?, seq_length, feature_size]
        #keep_prob_list : [?, seq_length, 1]
        self._seq_length = rnn_input.shape[1]
        self._NONE = tf.shape(rnn_input)[0]
        keep_prob_list = tf.reduce_min(keep_prob_list, axis=1) # [?, 1]
        # keep_prob_list = tf.transpose(keep_prob_list, [1, 0, 2]) # [seq_length, ?, 1]
        # keep_prob_list = tf.reshape(keep_prob_list, [self._seq_length, -1])
        #mask generation

        def mask_gen(self, val):
            def distrib(x):
                if x != 1:
                    mask = tf.cast((tf.contrib.distributions.Bernoulli(probs=x)).sample([self._hidden_size]), tf.float32)
                    mask = tf.reshape(mask, [-1, self._hidden_size])
                    mask = tf.tile(mask, [100, 1])
                    return mask
                else:
                    return tf.scalar_mul(0.75, tf.ones([100, 10]))
            mask_list = distrib(val)
            # mask_list = tf.map_fn(lambda x: distrib(x), list)
            return mask_list # ( 100, 10 )
        masks = tf.map_fn(lambda x: mask_gen(self, x), keep_prob_list)
        masks = tf.transpose(masks, [1,0,2])


        zeros_dims = tf.stack([2, tf.shape(rnn_input)[0], self._hidden_size])
        init_state = tf.fill(zeros_dims, 0.0)

        rnn_input = tf.transpose(rnn_input, [1, 0, 2])
        concat_input = tf.concat([rnn_input, masks], axis=2) # [ seq_length, ?, f + h]

        # scan은 반드시 init_state의 dimension 과 fn(step)의 리턴값의 dimension 이 동일해야한다.
        # scan의 initializer term은 step의 첫번째 인수에 해당한다.
        # transpose to (seq_length, ?, feature_size)
        h_and_c = tf.scan(self.step, concat_input, initializer=init_state)

        # need to transpose (seq_length, 2, ?, feature_size) -> (2, ?, seq_length, feature_size)
        hidden_states, cell_states = tf.unstack(tf.transpose(h_and_c, [1, 2, 0, 3]))

        return hidden_states, cell_states

    # def __call__(self, rnn_input): # no keep_prob_list
    #     # rnn input : [?, seq_length, hidden_size]
    #     zeros_dims = tf.stack([2, tf.shape(rnn_input)[0], self._hidden_size])
    #     init_state = tf.fill(zeros_dims, 0.0)
    #     self._seq_length = rnn_input.shape[1]
    #
    #     # scan은 반드시 init_state의 dimension 과 fn(step)의 리턴값의 dimension 이 동일해야한다.
    #     # scan의 initializer term은 step의 첫번째 인수에 해당한다.
    #     # transpose to (seq_length, ?, feature_size)
    #     h_and_c = tf.scan(self.step, tf.transpose(rnn_input, [1, 0, 2]), initializer=init_state)
    #
    #     # need to transpose (seq_length, 2, ?, feature_size) -> (2, ?, seq_length, feature_size)
    #     hidden_states, cell_states = tf.unstack(tf.transpose(h_and_c, [1, 2, 0, 3]))
    #
    #     return hidden_states, cell_states

    # def __call__(self, rnn_input): ''' peephole 상태라서 후에 쓰려면 수정해야함 weight에서는 peephole관련 삭제됨'''
    #     # [?, seq_length, hidden_size]
    #     zeros_dims = tf.stack([tf.shape(rnn_input)[0], self._hidden_size])
    #     init_state = tf.fill(zeros_dims, 0.0)
    #     self._seq_length = rnn_input.shape[1]
    #     self.hiddens = list(np.zeros(self._seq_length, dtype=np.object)) #리스트일 때, reshape가능
    #     self.hiddens[-1] = init_state
    #
    #     self.cell_states = list(np.zeros(self._seq_length, dtype=np.object))
    #     self.cell_states[-1] = init_state
    #
    #     # (?, seq_length, feature size) -> (?, feature size) X seq_length
    #     self.unstacked_inputs = tf.unstack(rnn_input, axis=1)
    #     # self.unstacked_inputs = tf.split(rnn_input, num_or_size_splits=self._seq_length, axis = 1)
    #     for t, input_t in enumerate(self.unstacked_inputs):
    #         '''
    #         concat_x : ( None, f + h + h )
    #         '''
    #         concat_x = tf.concat([input_t, self.hiddens[t-1], self.cell_states[t-1]], axis=1)
    #         ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
    #         # concat_x2 = concat_x[:, :(self._feature_size + self._hidden_size)]
    #         # concat_x2 = tf.concat([input_t, self.hiddens[t-1]], axis=1)
    #         cu = _my_lstm_cell.cell_update(self, concat_x[:, :(self._feature_size + self._hidden_size)])
    #
    #         cell_state_t = tf.multiply(fg, self.cell_states[t-1]) + tf.multiply(ig, cu)
    #         concat_x_t = tf.concat([input_t, self.hiddens[t-1], cell_state_t], axis=1)
    #         og = _my_lstm_cell.ou_gate(self, concat_x_t)
    #
    #         hidden_t = tf.multiply(og, tf.tanh(cell_state_t))
    #         self.hiddens[t] = hidden_t
    #         self.cell_states[t] = cell_state_t
    #
    #     self.hiddens = tf.reshape(self.hiddens, [-1 , self._seq_length, self._hidden_size])
    #     return self.hiddens, self.cell_states

    # def __call__(self, rnn_input):
    #     zeros_dims = tf.stack([tf.shape(rnn_input)[0], self._hidden_size])
    #     init_state = tf.fill(zeros_dims, 0.0)
    #     self.hiddens = np.zeros(rnn_input.shape[1], dtype=np.object)
    #     self.hiddens[-1] = init_state
    #
    #     self.cell_states = np.zeros(rnn_input.shape[1], dtype=np.object)
    #     self.cell_states[-1] = init_state
    #
    #     # (?, seq_length, feature size) -> (?, feature size) X seq_length
    #     self.unstacked_inputs = tf.unstack(rnn_input, axis=1)
    #
    #     for t, input_t in enumerate(self.unstacked_inputs):
    #         '''
    #         concat_x : ( None, f + h + h )
    #         '''
    #         concat_x = tf.concat([input_t, self.hiddens[t-1], self.cell_states[t-1]], axis=1)
    #         ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
    #         # concat_x2 = concat_x[:, :(self._feature_size + self._hidden_size)]
    #         # concat_x2 = tf.concat([input_t, self.hiddens[t-1]], axis=1)
    #         cu = _my_lstm_cell.cell_update(self, concat_x[:, :(self._feature_size + self._hidden_size)])
    #
    #         cell_state_t = tf.multiply(fg, self.cell_states[t-1]) + tf.multiply(ig, cu)
    #         concat_x_t = tf.concat([input_t, self.hiddens[t-1], cell_state_t], axis=1)
    #         og = _my_lstm_cell.ou_gate(self, concat_x_t)
    #
    #         hidden_t = tf.multiply(og, tf.tanh(cell_state_t))
    #         self.hiddens[t] = hidden_t
    #         self.cell_states[t] = cell_state_t
    #
    #     return self.hiddens, self.cell_states

# X = tf.placeholder(tf.float32, [None, 100, 3], name='x_input')
#
# lstm_cell = _my_lstm_cell(3, 10)
# hs, cs = lstm_cell(X)



