import numpy as np
import math
from functools import reduce

class SNNConv2d():
    def __init__(self, in_channels, out_channels, final_size, input_size, next_size, kernal_size, stride, padding, k1, k2, tau, vth, beta, lr, mode='dfa'):
        '''
        :param in_channels:
        :param out_channels:
        :param final_size:
        :param input_size:
        :param next_size:
        :param kernal_size:
        :param stride:
        :param padding:
        :param k1:
        :param k2:
        :param tau:
        :param vth:
        :param beta:
        :param lr:
        :param mode:
        '''
        # MSRA init
        # weight_scale = math.sqrt(reduce(lambda x, y: x * y, self.input_shape) / self.out_channels) # scale of weight init
        # self.weight = np.random.uniform(-1, 1, [self.kernal_size, self.kernal_size, self.in_channels, self.out_channels]) * k1
        # self.b = np.random.uniform(-1, 1, [self.output_size*self.output_size*self.out_channels, self.final_features]) * k2
        # self.b_rfa = np.random.uniform(-1, 1, [self.output_size*self.output_size*self.out_channels, self.final_features]) * k2

        # new generation
        # spike ! spike ! spike !
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.next_size = next_size
        self.final_size = final_size
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        self.output_size = (self.input_size + 2*self.padding - self.kernal_size)//self.stride + 1
        self.conv_cnt = (self.input_size - self.kernal_size)//self.stride # todo vital
        self.k1 = k1
        self.k2 = k2
        self.tau = tau
        self.vth = vth
        self.bate = beta
        self.lr = lr
        self.voltage = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.tmax = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.vmax = np.zeros((self.output_size, self.output_size, self.out_channels))

        # todo todo
        self.weight = np.random.uniform(-1, 1, (self.kernal_size, self.kernal_size, self.in_channels, self.out_channels)) * k1 # 这个权重初始化这里问题很大
        # todo todo

        if mode is 'dfa':
            self.b = np.random.uniform(-1, 1, [self.output_size*self.output_size*self.out_channels, final_size]) * k2
        elif mode is 'rfa':
            self.b = np.random.uniform(-1, 1, [self.output_size*self.output_size*self.out_channels, next_size]) * k2

        self.error = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.input_spike_state = np.zeros((self.input_size, self.input_size, self.in_channels))
        self.output_spike_state = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.output_spike_cnt = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.fire_time = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.ti = 0
        self.wmax = 1
        self.wmin = -1

        # todo 关键：由于权值共享，所以trace也不需要这么多！！！
        # todo 关键：虽然权值共享，但是卷积核的每个元素 应该说是对应于多个突触？？？ 这样Vmax这么求？
        # todo 关键：接上一个问题，这时候可以一是求出每组对应的Vmax，然后求一个平均，或者说是每次都更新
        # todo 关键：我先直接用公式推推看！！！
        self.trace_array = np.zeros((self.in_channels, self.out_channels, self.conv_cnt, self.kernal_size, self.kernal_size))
        self.trace_array_save = np.zeros((self.in_channels, self.out_channels, self.kernal_size, self.kernal_size))
        # todo trace这里得大改！！！

        # self.trace_array = np.zeros((self.in_channels, self.input_size*self.input_size))
        # self.trace_array_save = np.zeros((self.out_channels, self.input_size*self.input_size, self.output_size*self.output_size))






    def forward(self, input_spike_state, timestamp=None):
        '''
        :param input_spike_state: shape: (input_size, input_size, in_channels)
        :param label:
        :param timestamp:
        :return:

        CONV PER TIMESTAMP
        here i need to set up an threshold
        '''
        # col_weights = self.weight.reshape([-1, self.out_channels])
        # col_input = input_spike_state[np.newaxis, :] # add an axis for func im2col
        # self.output_spike_state = np.zeros(self.output_shape)
        #
        # self.col_spike_state = im2col(col_input, self.kernal_size, self.stride)
        # self.output_spike_state = np.reshape(np.dot(self.col_spike_state, col_weights), self.output_shape)
        #
        # # threshold judge
        # self.output_spike_state[self.output_spike_state < self.threshold] = 0
        # self.output_spike_state[self.output_spike_state >= self.threshold] = 1

        # ....................................................................
        self.input_spike_state = input_spike_state
        self.t = timestamp

        exp_decay = np.exp(-1.0 * (self.t - self.ti) / self.tau)
        self.trace_array *= exp_decay
        self.voltage *= exp_decay
        # self.voltage += np.dot(self.input_spike_state, self.weight)
        col_weights = self.weight.reshape([-1, self.out_channels])
        col_input_spikes = input_spike_state[np.newaxis, :]
        self.col_voltage = im2col(col_input_spikes, self.kernal_size, self.stride)
        # 前向过程中权值还是共享了..
        self.voltage += np.reshape(np.dot(self.col_voltage, col_weights), self.voltage.shape) # update memberance potential

        # 这里reshape只是为了使得后面的循环不要嵌套那么多重，其实开销甚至增大了
        self.voltage = np.reshape(self.voltage, (-1))
        self.fire_time = np.reshape(self.fire_time, (-1))
        self.vmax = np.reshape(self.vmax, (-1))
        self.tmax = np.reshape(self.tmax, (-1))
        self.output_spike_state = np.reshape(self.output_spike_state, (-1))

        for n in range(self.output_size*self.output_size*self.out_channels):
            # Refractory Period:
            if self.fire_time[n] !=0 and self.t-self.fire_time[n] <= self.beta:
                self.voltage[n] = 0

            # Trace Save & Max- Update:
            if self.voltage[n] > self.vmax[n]:
                self.trace_array_save[] # todo Alert

                self.vmax[n] = self.voltage[n]
                self.tmax[n] = self.t

            # Output Defination:
            if self.voltage[n] > self.vth:
                self.output_spike_state[n] = 1

                self.voltage[n] = 0 # reset the memberance potential to 0 after it fires
                self.fire_time[n] = self.t

                # todo 这里可以加一个 part 以便构建只有Conv结构的SNN

        # Add Spike to Trace:
        self.trace_array_save = [] # todo Alert

        self.ti = self.t # update the ti

        return self.output_spike_state



    def calc_error(self, output_error=[], mode='dfa'):
        '''
        Output Error Calculate:
        :param output_error:
        :param mode:
        :return: self.error (shape: (self.output_size, self.output_size, self.out_channels))
        '''
        if mode is 'dfa':
            feedback_error = np.dot(self.b, output_error)
            self.error = feedback_error.reshape((self.error.shape))
        elif mode is 'rfa':
            feedback_error = np.dot(self.b, output_error)
            self.error = feedback_error.reshape((self.error.shape))
        return self.error


    def backward(self, lr=0): # alpha=1e-3, weight_decay=0.0004
        '''
        :param alpha:
        :param weight_decay:
        :return:
        '''
        # gradient calculate // 公式看我pdf !!!TODO!!!
        error = np.reshape(self.error, [-1, self.out_channels])
        self.w_gradient = np.dot(self.col_spike_state.T, error).reshape(self.weight.shape)
        self.weight *= (1 - weight_decay)
        self.weight -= alpha * self.w_gradient

        #..............................................
        if lr==0: lr=self.lr
        self.trace_array_save[self.trace_array_save == 1] = 0
        delta_w = lr * self.trace_array_save * self.error #
        delta_w = np.dot(self.col_voltage.T, error).reshape(self.weight.shape)
        '''
        关键公式比较：
        '''





        return self.weight

    def load_w_and_b(self, name_save_path, name_w, name_b):
        '''
        :param name_save_path: 
        :param name_w: 
        :param name_b: 
        :return: 
        '''
        self.weight = np.load(name_save_path + name_w)
        self.b = np.load(name_save_path + name_b)
        
    def sava_w_and_b(self, name_save_path, name_w, name_b):
        '''
        :param name_save_path:
        :param name_w:
        :param name_b:
        :return:
        '''
        np.save(name_save_path + name_w, self.weight)
        np.save(name_save_path + name_b, self.b)



# In order to accelerate conv
# todo under modification

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


if __name__ == "__main__":
    out_error = np.random.randint(0, 2, (200,))
    img = np.random.randint(0, 2, (16,16,1))
    conv = SNNConv2d(in_channels=1, out_channels=3, final_features=200, input_size=16, kernal_size=3, stride=1, k2=1, threshold=0.25)
    next = conv.forward(img, 0, 0)
    error = conv.calc_error(out_error)
    print('pre weight is:', conv.weight[:,:,0,0])
    weight = conv.backward()
    # print('max is:', np.max(next))
    # print('shape is:', np.shape(next))
    print('weight is:', weight[:,:,0,0])
    print(next)