import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\data')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\layers')))
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data.mnist_dataset import load
from layers.coding import SNNInput
from layers.conv import SNNConv2d
from layers.linear import SNNLinear
from layers.pool import SNNMaxPooling


if __name__ == "__main__":

    '''
    明天早上先对一层卷积层进行试验！！！
    
    '''


    
    weight = 'weight.npy'
    random_feedback_matrix_b = 'b.npy'
    
    # hyper parameters:
    # network_architecture = [5*5*32, 500, 500, 200]
    network_architecture = [5*5*32, 500, 200]
    vth = [5, 15, 15]
    lr = 1e-3
    beta = 3
    tau = 1000
    k2 = 0.1

    # data process:
    data = load()
    train_set = data.train
    validation_set = data.validation
    test_set = data.test
    train_acc = []
    validation_acc = []
    test_acc = []

    # network structure:
    input_layer = SNNInput(train_set, validation_set, test_set)
    # 把input_size那个改了，不太智能
    conv_1 = SNNConv2d(in_channels=1, out_channels=16, final_features=200, input_size=28, kernal_size=5, stride=1, k2=1, threshold=1)
    pool_1 = SNNMaxPooling(shape=[24, 24, 16], kernal_size=2, stride=2)
    conv_2 = SNNConv2d(in_channels=16, out_channels=32, final_features=200, input_size=12, kernal_size=3, stride=1, k2=1, threshold=1)
    pool_2 = SNNMaxPooling(shape=[10, 10, 32], kernal_size=2, stride=2)
    hidden_1 = SNNLinear(network_architecture[0], network_architecture[1], network_architecture[-1], k2, tau, vth[0], beta, lr)
    # hidden_2 = SNNLinear(network_architecture[1], network_architecture[2], network_architecture[-1], k2, tau, vth[1], beta, lr)
    output_layer = SNNLinear(network_architecture[1], network_architecture[2], network_architecture[-1], k2, tau, vth[1], beta, lr, class_num = 10)

    # paras load:
    # hidden_1.load_w_and_b('C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\paras\\', 'hidden_1_weight.npy', 'hidden_1_b.npy')
    output_layer.load_w_and_b('C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\paras\\', 'output_weight.npy', 'output_b.npy')

    # spike encoding:
    train_spike, validation_spike, test_spike = input_layer.temporal_encoding()
    
    # train:
    start_train = time.time()  # 训练计时
    for i in range(1000):
        for t in range(255): # or it can be 256
            input_spike_state, input_img_label = input_layer.forward(i, t)
            if np.sum(input_spike_state) != 0:
                input_spike_state = np.reshape(input_spike_state, (28, 28, 1))
                output_spike_state = conv_1.forward(input_spike_state)
                output_spike_state = pool_1.forward(output_spike_state)
                output_spike_state = conv_2.forward(output_spike_state)
                output_spike_state = pool_2.forward(output_spike_state)

                reshape_to_flatten = output_spike_state.shape
                out_flatten = output_spike_state.reshape(-1)

                output_spike_state, valid_train = hidden_1.forward(out_flatten, input_img_label, t)
                # output_spike_state, _ = hidden_2.forward(output_spike_state, input_img_label, t)
                output_spike_state, _ = output_layer.forward(output_spike_state, input_img_label, t)

        output_error = output_layer.calc_error()
        # hidden_2.calc_error(output_error)
        hidden_1.calc_error(output_error)
        conv_2.calc_error(output_error)
        conv_1.calc_error(output_error)

        output_layer.backward(lr)
        # hidden_2.backward(lr)
        hidden_1.backward(lr)
        conv_2.backward()
        conv_1.backward()

        predict_ans = output_layer.class_result()
        train_acc.append(predict_ans)
        
        hidden_1.clear()
        # hidden_2.clear()
        output_layer.clear()

    end_train = time.time()
    train_acc = np.array(train_acc)
    right = len(np.where(train_acc==1)[0]) # 这玩意可能有问题！！！
    wrong = len(np.where(train_acc==0)[0])
    unkown = len(np.where(train_acc==-1)[0])

    print('train acc ratio is: ', right/np.shape(train_acc)[0], '& unkown ratio: ', unkown / np.shape(train_acc)[0], '\n', '->train time is: ', (end_train-start_train)/60, 'mins')

    # test:
    start_test = time.time()  # 测试计时
    for i in range(200):
        for t in range(255):
            input_spike_state, input_img_label = input_layer.forward(i, t, mode='test')
            if np.sum(input_spike_state) != 0:
                output_spike_state = conv_1.forward(input_spike_state)
                output_spike_state = pool_1.forward(output_spike_state)
                output_spike_state = conv_2.forward(output_spike_state)
                output_spike_state = pool_2.forward(output_spike_state)

                reshape_to_flatten = output_spike_state.shape
                out_flatten = output_spike_state.reshape(-1)

                output_spike_state, valid_train = hidden_1.forward(out_flatten, input_img_label, t)
                # output_spike_state, _ = hidden_2.forward(output_spike_state, input_img_label, t)
                output_spike_state, _ = output_layer.forward(output_spike_state, input_img_label, t)

        predict_ans = output_layer.class_result()
        test_acc.append(predict_ans)

        hidden_1.clear()
        # hidden_2.clear()
        output_layer.clear()
    end_test = time.time()

    test_acc = np.array(test_acc)
    right = len(np.where(test_acc==1)[0])
    wrong = len(np.where(test_acc==0)[0])
    unkown = len(np.where(test_acc==-1)[0])
    print('test acc ratio is: ', right / np.shape(test_acc)[0], '& unkown ratio: ', unkown / np.shape(test_acc)[0], '\n', '->test time is: ', (end_test-start_test)/60, 'mins')
