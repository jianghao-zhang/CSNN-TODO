# Protected by Zhang JiangHao
import numpy as np
from utils.init_methods import initialize_parameters_he
# 这个类需要包含的功能：
# 计算出往下一层传输的那个t-x-2 tensor
# update两层之间的weight，
# 输入层确实是位置与发射时间，但后面每层的输入和输出在某个特定时间步都只是一个位置+发射与否的向量

class SNNLinear():
    def __init__(self, in_features, out_features, final_features, k2, tau, vth, beta, lr, class_num = 0, mode = 'dfa', next_features=0):
        '''
        Input Parameter Explain:: 
            in_features: size of inputs'
            out_features: size of outputs'
            final_features: size of output layer over nn
            k2: init para of fixed random feedback matrix, layers up->k2 should done
            class_num: the number of classes of the dataset
        Output Parameter Explain:
            self.weight: weight between layer_out & layer_in
            self.b: fixed random feedback matrix
            self.trace_array: Tempotron trace
            self.trace_array_save:
            self.voltage: memberance potential 
            self.tmax: time that memberance potential reaches peak
            self.vmax: peak of memberance potential
            self.tau: tau->infinite <--> IF 
            self.vth: threshold
            self.beta: Refractory Period
            self.lr: Learning Rate
            self.fire_time: record fire time
            self.error: as its name
            self.input_spike_state: vector that contains info of pos & fire or not at a specific timestamp 
            self.output_spike_state: output of next layer
            self.output_spike_cnt: output_spikes cnt of 2nd layer
            self.output_spike_time: store the spike time & num of next layer
            self.ti: the time when a presynaptic spike arrives at synapse i
            self.tprev: the time of its immediate previous presynaptic spike arriving at any synapse
            self.class_num: the number of classes of the dataset; if it doesn't equal to 0, it means this layer is output layer
            self.pool_neu_num: num of neuron in single neuron pool
            self.pool_spike_num: in order to count the num of spikes of every class, in order to show train acc
        '''
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.final_features = final_features
        self.next_features = next_features
        self.weight = initialize_parameters_he([self.in_features, self.out_features]).T * 0.1 #  prev:0.1 for one conv one linear ; prev:3 for only conv
        if self.mode == 'dfa':
            self.b = np.random.uniform(-1, 1, [out_features, final_features]) * k2
        elif self.mode == 'rfa':
            self.b = np.random.uniform(-1, 1, [self.out_features, self.next_features]) * k2

        self.trace_array = np.zeros((in_features, ))
        self.trace_array_save = np.zeros((in_features, out_features))
        self.voltage = np.zeros((out_features, ))
        self.tmax = np.zeros((out_features, ))
        self.vmax = np.zeros((out_features, ))
        self.tau = tau
        self.vth = vth
        self.beta = beta
        self.lr = lr
        self.fire_time = np.zeros((out_features, ))
        self.error = np.zeros((out_features, ))
        self.input_spike_state = np.zeros((in_features, ))
        self.output_spike_state = np.zeros((out_features, ))
        self.output_spike_cnt = np.zeros((out_features, ))
        self.ti = 0
        self.class_num = class_num
        # Judge If It is Final Output Layer:
        if class_num != 0:
            self.pool_neu_num = int(self.out_features / class_num)
            self.pool_spike_state = np.zeros((self.final_features, ))
            self.pool_spike_num = np.zeros((class_num, ))
            
        #  upper & lower limit of Weight:
        self.wmax = 1
        self.wmin = -1
        # todo-todo-todo-todo-todo: 感觉这样下去不太行，得加一个正则化项

    # Spike Forwards
    def forward(self, input_spike_state, timestamp):
        '''
        :param input_spike_state: vector that contains info of pos & fire or not at a specific timestamp
        :param timestamp: time point
        :return: self.output_spike_state, valid_train
        '''
        valid_train = True
        self.input_spike_state = input_spike_state # save real time input spike state
        self.t = timestamp # save real time
        
        # 前面加一个判断此时刻是否无脉冲输入 todo todo todo
        if np.sum(input_spike_state) == 0:
            valid_train = False
            return self.output_spike_state, valid_train
        else: # 如果确实有脉冲输入
            # pre_spike_pos = np.argwhere(input_spike_state == 1) # 得到所有发射了脉冲的上一层神经元的位置
            exp_decay = np.exp(-1.0 * (self.t - self.ti) / self.tau) # 计算exp decay
            
            # 我知道了，这部分在硬件中是可以并行的, 那无所谓了，for循环上吧！！！
            self.trace_array *= exp_decay # damping of trace
            self.voltage *= exp_decay # damping of memberance potential
            self.voltage += np.dot(self.input_spike_state, self.weight) # update memberance potential
            
            for n in range(self.out_features):
                # Refractory Period:
                if self.fire_time[n] !=0 and self.t-self.fire_time[n] <= self.beta:
                    self.voltage[n] = 0
                
                # Trace Save & max- update:
                if self.voltage[n] > self.vmax[n]:
                    self.trace_array_save[:, n] = self.trace_array # all synapses inject into neuron n
                    self.vmax[n] = self.voltage[n]
                    self.tmax[n] = self.t
                
                # Output Defination:
                if self.voltage[n] > self.vth:
                    self.output_spike_state[n] = 1
                    self.output_spike_cnt[n] += 1
                    
                    self.voltage[n] = 0 # reset the memberance potential to 0 after it fires
                    self.fire_time[n] = self.t # record the fire time of neuron n in next layer
                    
                    # Judge If It is Final Output Layer:
                    if self.class_num != 0:
                        pool_num = int(np.floor(n/self.pool_neu_num)) # judge which pool it belongs to
                        self.pool_spike_num[pool_num] += 1
                        # self.pool_spike_state[n] = 1
                else:
                    self.output_spike_state[n] = 0

            # Add Spike to Trace:
            self.trace_array += self.input_spike_state

            self.ti = self.t # update the ti
            # if self.class_num != 0 and timestamp > 150:
            #     print(timestamp, '\n',self.output_spike_cnt)
        return self.output_spike_state, valid_train # self.pool_spike_num, self.output_spike_time


    # Calculate Error
    def calc_error(self, output_error=[], label=-1):
        '''
        Error Calculate:
        :para label: label of input image
        :param output_error: error of final output layer
        :return: self.error
        '''
        self.label = label
        if self.class_num != 0: # Output Layer
            for n in range(self.out_features):
                pool_num = int(np.floor(n / self.pool_neu_num))  # 该神经元属于哪个池
                if pool_num == self.label and self.output_spike_cnt[n] == 0: # should fire but not fire todo
                    self.error[n] = 1 # V_th[hidden_num] - Vmax_o[n] TODO
                elif pool_num != self.label and self.output_spike_cnt[n] > 0: # shouldn't fire but fire
                    self.error[n] = -1 # V_th - Vmax_o[n] TODO
                    '''
                    其实上面这个必然是可以改进的，like与脉冲发射数关联
                    '''

        else: # Hidden Layers
            self.error = np.dot(self.b, output_error) # directly feedback error
        return self.error
        
    # Update Weights
    def backward(self, lr):
        '''
        Update Weights by Trace
        :param lr: learning rate
        :return: self.weight
        '''
        self.trace_array_save[self.trace_array_save == 1] = 0
        delta_w = lr * self.trace_array_save * self.error # 这里是self.trace_array_save的每一行与self.error的每一行对应元素相乘

        self.weight += delta_w
        self.weight[self.weight>self.wmax] = self.wmax
        self.weight[self.weight<self.wmin] = self.wmin

        # todo-todo-todo: obverse
        # print('linear weight sum: ', np.sum(self.weight))
        return self.weight
    
    
    # Judge Class Success or not
    def class_result(self, label=-1):
        # Population Judge
        mx = np.where(self.pool_spike_num == np.max(self.pool_spike_num))
        mx_cnt = len(mx[0])

        if label != -1:
            self.label = label
        
        if mx_cnt == 1 and mx[0][0] == self.label:
            print('label is: ', self.label, ', predict is:', mx[0][0], '--> right', ', mx_cnt is: ', mx_cnt, ' pool_spike_num: ', self.pool_spike_num)
            return 1 # right type
        elif mx_cnt == 1 and mx[0][0] != self.label:
            print('label is: ', self.label, ', predict is:', mx[0][0], '--> wrong', ', mx_cnt is: ', mx_cnt, ' pool_spike_num: ', self.pool_spike_num)
            return 0 # wrong type
        elif mx_cnt >= 2 and not mx[0].__contains__(self.label):
            print('label is: ', self.label, ', predict is:', mx[0][0], '--> wrong', ', mx_cnt is: ', mx_cnt, ' pool_spike_num: ', self.pool_spike_num)
            return 0 # wrong type
        elif mx_cnt >= 2 and mx[0].__contains__(self.label):
            print('label is: ', self.label, ', predict is:', mx[0][0], '--> unkown', ', mx_cnt is: ', mx_cnt, ' pool_spike_num: ', self.pool_spike_num)
            return -1 # unknow type 

    
    # Clear Cache(for next image)
    def clear(self):
        self.trace_array = np.zeros((self.in_features, ))
        self.trace_array_save = np.zeros((self.in_features, self.out_features))
        self.voltage = np.zeros((self.out_features, ))
        self.tmax = np.zeros((self.out_features, ))
        self.vmax = np.zeros((self.out_features, ))
        self.fire_time = np.zeros((self.out_features, ))
        self.error = np.zeros((self.out_features, ))
        self.input_spike_state = np.zeros((self.in_features, ))
        self.output_spike_state = np.zeros((self.out_features, ))
        self.output_spike_cnt = np.zeros((self.out_features, ))
        self.ti = 0
        # Judge If It is Final Output Layer:
        if self.class_num != 0:
            self.pool_spike_num = np.zeros((self.class_num, ))
            self.pool_spike_state = np.zeros((self.final_features,))


    # Load Weights and Random Feedback Matrix b
    def load_w_and_b(self, name_save_path, name_w, name_b):
        '''
        :param name_save_path: name(str) of the path to save
        :param name_w: name(str) of the weight matrix to save
        :param name_b: name(str) of the random feedback matrix to save
        '''
        self.weight = np.load(name_save_path + name_w)
        self.b = np.load(name_save_path + name_b)


    # Save Weights and Random Feedback Matrix b
    def save_w_and_b(self, name_save_path, name_w, name_b):
        '''
        :param name_save_path: name(str) of the path to save
        :param name_w: name(str) of the weight matrix to save
        :param name_b: name(str) of the random feedback matrix to save
        '''
        np.save(name_save_path + name_w, self.weight)
        np.save(name_save_path + name_b, self.b)
        