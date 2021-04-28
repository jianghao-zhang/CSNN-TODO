import numpy as np
import math
from functools import reduce

'''
A failed version 
I've stopped experimenting with this version.
25/03/2021
'''


# 这个不行的话就试试rfa!!!! TODO TODO TODO
# 这个不行的话就试试rfa!!!! TODO TODO TODO


# 对batch是否可以对snn训练产生正面影响存疑
# 我觉得这里的卷积只是一种运算，不对，卷积层应该有膜电位,不然没办法拓展到多层卷积结构上
# 不对，不必有膜电位也ok的，我们只需要把卷积当成一种特殊的通道即可！！
# 对每个timestamp的脉冲发射state进行卷积
# 直接使用timestamp


class SNNConv2d():
    def __init__(self, in_channels, out_channels, final_features, input_size, kernal_size=3, stride=1, k1=1, k2=1, threshold=1):
        '''
        :param in_channels: Determine the number of channels of the convolution kernal
        :param out_channels: Determine the number of convolution kernals
        :param kernal_size:
        :param stride:
        :param init_params:

        改写成不管batch_size即可
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_features = final_features
        self.input_size = input_size
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = 0
        self.threshold = threshold
        self.output_size = (self.input_size + 2 * self.padding - self.kernal_size) // self.stride + 1

        self.input_shape = np.array([self.input_size, self.input_size, self.in_channels])
        self.output_shape = np.array([self.output_size, self.output_size, self.out_channels])

        # MSRA init
        weight_scale = math.sqrt(
            reduce(lambda x, y: x * y, self.input_shape) / self.out_channels)  # scale of weight init
        # todo 这个weight scale怕是要改
        self.weight = np.random.uniform(-1, 1,
                                        [self.kernal_size, self.kernal_size, self.in_channels, self.out_channels]) * k1
        self.b = np.random.uniform(-1, 1,
                                   [self.output_size * self.output_size * self.out_channels, self.final_features]) * k2

        self.b_rfa = np.random.uniform(-1, 1, [self.output_size * self.output_size * self.out_channels,
                                               self.final_features]) * k2

        print('conv weight is: ', np.sum(self.weight))
        print('b is: ', np.sum(self.b))
        print('conv ratio is: ', np.sum(self.weight) / np.sum(self.b))

        # 后面要reshape回来的！！

    def forward(self, input_spike_state, label=None, timestamp=None):
        '''
        :param input_spike_state: shape: (input_size, input_size, in_channels)
        :param label:
        :param timestamp:
        :return:

        CONV PER TIMESTAMP
        here i need to set up an threshold
        '''
        col_weights = self.weight.reshape([-1, self.out_channels])
        col_input = input_spike_state[np.newaxis, :]  # add an axis for func im2col
        self.output_spike_state = np.zeros(self.output_shape)

        self.col_spike_state = im2col(col_input, self.kernal_size, self.stride)
        self.output_spike_state = np.reshape(np.dot(self.col_spike_state, col_weights), self.output_shape)

        # threshold judge
        self.output_spike_state[self.output_spike_state < self.threshold] = 0
        self.output_spike_state[self.output_spike_state >= self.threshold] = 1

        return self.output_spike_state

    def forward_spike(self, input_spike_state, label=None, timestamp=None):
        '''
        :param input_spike_state:
        :param label:
        :param timestamp:
        :return:
        '''
        # 改成膜电位累计式卷积，并且加上脉冲
        self.label = label
        self.input_spike_state = input_spike_state
        self.t = timestamp

        exp_decay = np.exp(-1.0 * (self.t - self.ti) / self.tau)
        self.trace_array *= exp_decay
        self.voltage *= exp_decay
        self.voltage +=

        for n in range(self.out_features)

        pass

    def calc_error(self, output_error=[]):
        '''
        :param output_error:
        :return:
        '''
        # Output Error Calculate:
        if mode is 'dfa':
            self.error = np.dot(self.b, output_error)
        elif mode is 'rfa':
            self.error =
        return self.error

    def backward(self, alpha=1e-3, weight_decay=0.0004):
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
    img = np.random.randint(0, 2, (16, 16, 1))
    conv = SNNConv2d(in_channels=1, out_channels=3, final_features=200, input_size=16, kernal_size=3, stride=1, k2=1,
                     threshold=0.25)
    next = conv.forward(img, 0, 0)
    error = conv.calc_error(out_error)
    print('pre weight is:', conv.weight[:, :, 0, 0])
    weight = conv.backward()
    # print('max is:', np.max(next))
    # print('shape is:', np.shape(next))
    print('weight is:', weight[:, :, 0, 0])
    print(next)
