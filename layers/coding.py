import numpy as np
# 要编码成 --> 像素位置+脉冲时间
class SNNInput():
    def __init__(self, train_set, validation_set, test_set):
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.train_spike = []
        self.validation_spike = []
        self.test_spike = []
        self.seq_train = []
        self.seq_validation = []
        self.seq_test = []
        self.ti = 0

        
    def temporal_encoding(self, k=1):
        # in order to calculate the spike times -- ti
        # k is a constant which controls the temporal encoding's scale
        # output shape: e.g. train: 60000, 2 --> [x][0]:2,784 & [x][1]:label
        train_num = np.shape(self.train_set[1])[0]
        validation_num = np.shape(self.validation_set[1])[0]
        test_num = np.shape(self.test_set[1])[0]
        
        self.seq_train = np.arange(train_num)
        self.seq_validation = np.arange(validation_num)
        self.seq_test = np.arange(test_num)
        
        temp_pos = np.arange(np.shape(self.train_set[0])[2] * np.shape(self.train_set[0])[3])
        
        temp_time_train = list(map(lambda x: k*(255-np.reshape(x, (784,))), self.train_set[0]))
        for i in range(train_num):
            temp = []
            temp.append(np.vstack((temp_pos, temp_time_train[i])))
            temp.append(self.train_set[1][i])
            self.train_spike.append(temp)

        temp_time_validation = list(map(lambda x: k*(255-np.reshape(x, (784,))), self.validation_set[0]))
        for i in range(validation_num):
            temp = []
            temp.append(np.vstack((temp_pos, temp_time_validation[i])))
            temp.append(self.validation_set[1][i])
            self.validation_spike.append(temp)

        temp_time_test = list(map(lambda x: k*(255-np.reshape(x, (784,))), self.test_set[0]))
        for i in range(test_num):
            temp = []
            temp.append(np.vstack((temp_pos, temp_time_test[i])))
            temp.append(self.test_set[1][i])
            self.test_spike.append(temp)
        
        return self.train_spike, self.validation_spike, self.test_spike
    
    def rate_encoding(self):
        pass
    
    def forward(self, img_num, timestamp, mode='train'):
        # timestamp范围：0~254 , 我觉得不去管255是一个更好的选择 todo todo
        if mode is 'train':
            spike_set = self.train_spike
            n = self.seq_train[img_num]
            # n = img_num
        elif mode is 'validation':
            spike_set = self.validation_spike
            n = self.seq_validation[img_num]
            # n = img_num
        elif mode is 'test':
            spike_set = self.test_spike
            n = self.seq_test[img_num]
            # n = img_num

        input_img = spike_set[n][0]
        input_img_label = spike_set[n][1]
        input_spike_state = np.zeros((np.shape(self.train_set[0])[2] * np.shape(self.train_set[0])[3], ))

        img = np.array(input_img[1])
        spike_pos = np.where(img == timestamp)[0]
        if len(spike_pos) == 0:
            return input_spike_state, input_img_label
        else:
            input_spike_state = img.copy()
            input_spike_state[input_spike_state != timestamp] = -1
            input_spike_state[input_spike_state == timestamp] = -2
            input_spike_state[input_spike_state == -1] = 0
            input_spike_state[input_spike_state == -2] = 1

            return input_spike_state, input_img_label
