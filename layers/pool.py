import numpy as np

class SNNMaxPooling():
    def __init__(self, shape, kernal_size=2, stride=2):
        '''
        :param shape:
        :param kernal_size:
        :param stride:
        '''
        self.out_channels = shape[-1]
        self.kernal_size = kernal_size
        self.stride = stride
        self.index = np.zeros(shape)
        self.output_shape = [shape[0]//self.stride, shape[1]//self.stride, self.out_channels]

    def forward(self, input_spike_state):
        '''
        :param input_spike_state:
        :return:
        '''
        self.output_spike_state = np.zeros([input_spike_state.shape[0]//self.stride, input_spike_state.shape[1]//self.stride, self.out_channels])

        for c in range(self.out_channels):
            for i in range(0, input_spike_state.shape[0], self.stride):
                for j in range(0, input_spike_state.shape[1], self.stride):
                    self.output_spike_state[i//self.stride, j//self.stride, c] = np.max(input_spike_state[i:i+self.kernal_size, j:j+self.kernal_size, c])
                    index = np.argmax(input_spike_state[i:i+self.kernal_size, j:j+self.kernal_size, c])
                    self.index[i+index//self.stride, j+index//self.stride, c] = 1
        return self.output_spike_state

    def calc_error(self, output_error=[]):
        pass

    def backward(self):
        pass

    def save_w_and_b(self):
        pass


if __name__ == "__main__":

    img = np.random.randint(0, 2, (32,32,1))
    pool = SNNMaxPooling(img.shape, 2, 2)
    img1 = pool.forward(img)
    print(img[:,:,0])
    print(img1[:,:,0])

