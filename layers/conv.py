

import numpy as np

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
        self.output_size = (self.input_size + 2 * self.padding - self.kernal_size) // self.stride + 1
        self.conv_cnt = (self.input_size - self.kernal_size) // self.stride  # todo vital
        self.k1 = k1
        self.k2 = k2
        self.tau = tau
        self.vth = vth
        self.beta = beta
        self.lr = lr
        self.voltage = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.tmax = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.vmax = np.zeros((self.output_size, self.output_size, self.out_channels))

        self.weight = np.random.uniform(-1, 1, (self.kernal_size, self.kernal_size, self.in_channels, self.out_channels)) * k1  # 这个权重初始化这里问题很大

        # features should be checked in formula.goodnote
        if mode == 'dfa':
            self.b = np.random.uniform(-1, 1, [self.output_size * self.output_size, final_size, self.out_channels]) * k2
        elif mode == 'rfa':
            self.b = np.random.uniform(-1, 1, [self.output_size * self.output_size, next_size, self.out_channels]) * k2

        self.error = np.zeros((self.output_size, self.output_size, self.out_channels))

        self.input_spike_state = np.zeros((self.input_size, self.input_size, self.in_channels))
        self.output_spike_state = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.output_spike_cnt = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.fire_time = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.t = 0
        self.ti = 0
        self.wmax = 1
        self.wmin = -1

        # todo-回归trace,回归本源, 用来存储exp_decay的积*R, 但这个也不能叫trace了，(*^_^*)
        # todo-内存占用比较大啊
        self.trace = np.zeros((self.kernal_size, self.kernal_size, self.in_channels, self.output_size, self.output_size, self.out_channels))
        self.trace = np.reshape(self.trace, (self.kernal_size, self.kernal_size, self.in_channels, -1))

    # todo
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
        NoneInput = False
        self.input_spike_state = input_spike_state

        if np.sum(self.input_spike_state) == 0:
            NoneInput = True
            return NoneInput, self.output_spike_state
        else:
            self.t = timestamp
            exp_decay = np.exp(-1.0 * (self.t - self.ti) / self.tau)
            self.voltage *= exp_decay

            col_weights = self.weight.reshape([-1, self.out_channels])
            col_input_spikes = input_spike_state[np.newaxis, :]
            self.col_voltage = im2col(col_input_spikes, self.kernal_size, self.stride)
            # todo sibalaxi deshu forward path, 这个前向的过程应该是没有问题的
            self.voltage += np.reshape(np.dot(self.col_voltage, col_weights), self.voltage.shape)  # update memberance potential

            # 这里reshape只是为了使得后面的循环不要嵌套那么多重，其实开销甚至增大了
            # todo-todo-下面这块可以优化一下，没必要每次都计算的
            self.voltage = np.reshape(self.voltage, (-1))
            self.fire_time = np.reshape(self.fire_time, (-1))
            self.vmax = np.reshape(self.vmax, (-1))
            self.tmax = np.reshape(self.tmax, (-1))
            self.output_spike_state = np.reshape(self.output_spike_state, (-1))

            for n in range(self.output_size * self.output_size * self.out_channels):
                # Refractory Period:
                if self.fire_time[n] != 0 and self.t - self.fire_time[n] <= self.beta:
                    self.voltage[n] = 0

                # Trace Save & Max- Update:
                if self.voltage[n] > self.vmax[n]:
                    self.vmax[n] = self.voltage[n]
                    self.tmax[n] = self.t
                    self.trace[:, :, :, n] *= exp_decay

                    # todo-得到n原来的位置
                    # index_c = n // (self.output_size * self.output_size)
                    index_row = (n % (self.output_size * self.output_size)) // self.output_size
                    index_column = (n % (self.output_size * self.output_size)) % self.output_size

                    # todo-找到对应的Receptive-field并且加上去
                    self.trace[:, :, :, n] += self.input_spike_state[index_row * self.stride:index_row * self.stride + self.kernal_size, index_column * self.stride:index_column * self.stride + self.kernal_size,:]

                # Output Defination:
                if self.voltage[n] > self.vth:
                    self.output_spike_state[n] = 1
                    self.voltage[n] = 0  # reset the memberance potential to 0 after it fires
                    self.fire_time[n] = self.t
                else:
                    self.output_spike_state[n] = 0

                    # todo 这里可以加一个 part 以便构建只有Conv结构的SNN, 从ann没有linear层也可以表现得ok看，这个应该也行

            # todo-得把某些不必要的东西reshape回来
            self.output_spike_state = np.reshape(self.output_spike_state, (self.output_size, self.output_size, self.out_channels))

            self.ti = self.t  # update the ti

        return NoneInput, self.output_spike_state

    # done
    def calc_error(self, output_error=[], mode='dfa'):
        '''
        Output Error Calculate:
        :param output_error:
        :param mode:
        :return: self.error (shape: (self.output_size, self.output_size, self.out_channels))
        '''
        # todo-todo-todo-其实传回来的不是error，而是polarity！！！
        if mode == 'dfa':
            for i in range(self.out_channels):
                self.error[:, :, i] = np.dot(self.b[:, :, i], output_error).reshape((self.output_size, self.output_size))
        elif mode == 'rfa':
            for i in range(self.out_channels):
                self.error[:, :, i] = np.dot(self.b[:, :, i], output_error).reshape((self.output_size, self.output_size))
        return self.error

    # todo-最后一块拼图了
    def backward(self, lr=0):  # alpha=1e-3, weight_decay=0.0004
        '''
        :param alpha:
        :param weight_decay:
        :return:
        '''
        if lr == 0: lr = self.lr
        weight_gradient = np.zeros((self.kernal_size, self.kernal_size, self.in_channels, self.out_channels))
        temp_sum = [[] for i in range(self.out_channels)]
        polarity_sum = [np.sum(self.error[:, :, i]) for i in range(self.out_channels)]

        for n in range(self.output_size * self.output_size * self.out_channels):
            index_c = n // (self.output_size * self.output_size)
            if temp_sum[index_c] == []:
                temp_sum[index_c] = self.trace[:, :, :, n]
            else:
                temp_sum[index_c] += self.trace[:, :, :, n]

        for c in range(self.out_channels):
            weight_gradient[:, :, :, c] = temp_sum[c] * polarity_sum[c] * lr

        # todo-update weights:
        weight_gradient = np.array(weight_gradient)
        self.weight += weight_gradient

        # todo-limits:
        self.weight[self.weight > self.wmax] = self.wmax
        self.weight[self.weight < self.wmin] = self.wmin

        # print('conv weight sum is:', np.sum(self.weight))
        return self.weight


    # todo-todo-todo-clear:
    def clear(self):
        self.input_spike_state = np.zeros((self.input_size, self.input_size, self.in_channels))
        self.output_spike_state = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.output_spike_cnt = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.fire_time = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.voltage = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.tmax = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.vmax = np.zeros((self.output_size, self.output_size, self.out_channels))
        self.trace = np.zeros((self.kernal_size, self.kernal_size, self.in_channels, self.output_size, self.output_size, self.out_channels))
        self.trace = np.reshape(self.trace, (self.kernal_size, self.kernal_size, self.in_channels, -1))
        self.ti = 0
        self.t = 0


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

# if __name__ == "__main__":
# out_error = np.random.randint(0, 2, (200,))
# img = np.random.randint(0, 2, (16,16,1))
# conv = SNNConv2d(in_channels=1, out_channels=3, final_features=200, input_size=16, kernal_size=3, stride=1, k2=1, threshold=0.25)
# next = conv.forward(img, 0, 0)
# error = conv.calc_error(out_error)
# print('pre weight is:', conv.weight[:,:,0,0])
# weight = conv.backward()
# # print('max is:', np.max(next))
# # print('shape is:', np.shape(next))
# print('weight is:', weight[:,:,0,0])
# print(next)
