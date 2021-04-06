import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), 'C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\data')))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), 'C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\layers')))

import time
import numpy as np
import matplotlib.pyplot as plt
from data.mnist_dataset import load
from layers.linear import SNNLinear
from layers.coding import SNNInput

if __name__ == "__main__":

    weight = 'weight.npy'
    random_feedback_matrix_b = 'b.npy'

    # hyper parameters:
    network_architecture = [784, 500, 500, 100]
    # network_architecture = [784, 500, 100]
    vth = [5, 15, 15]
    lr = 1e-4
    beta = 3
    tau = 500
    k2 = 0.05

    # vth = [5, 15, 15]
    # lr = 1e-4
    # beta = 3
    # tau = 500
    # k2 = 0.05

    iteration = 4

    # data process:
    data = load()
    train_set = data.train
    validation_set = data.validation
    test_set = data.test
    train_acc = []
    validation_acc = []
    test_acc = []

    # 这里还要加一下那个


    # network structure:
    input_layer = SNNInput(train_set, validation_set, test_set)
    hidden_1 = SNNLinear(network_architecture[0], network_architecture[1], network_architecture[-1], k2, tau, vth[0], beta, lr, mode='rfa', next_features=network_architecture[2])
    hidden_2 = SNNLinear(network_architecture[1], network_architecture[2], network_architecture[-1], k2, tau, vth[1], beta, lr, mode='rfa', next_features=network_architecture[3])
    output_layer = SNNLinear(network_architecture[2], network_architecture[3], network_architecture[-1], k2, tau, vth[2], beta, lr, class_num = 10, mode='rfa')
    # output_layer = SNNLinear(network_architecture[1], network_architecture[2], network_architecture[-1], k2, tau, vth[1], beta, lr, class_num=10, mode='rfa')

    # paras load:
    # hidden_1.load_w_and_b('C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\paras\\rfa\\linear_3_layers\\4times\\', 'weight_h1.npy', 'b_h1.npy')
    # output_layer.load_w_and_b('C:\\Users\\Alisa\\Desktop\\CSNN_Project\\ToSNN\\CSNN\\paras\\rfa\\linear_3_layers\\4times\\', 'weight_o.npy', 'b_o.npy')

    # spike encoding:
    train_spike, validation_spike, test_spike = input_layer.temporal_encoding()
    # train:
    for ite in range(iteration):
        start_train = time.time()  # 训练计时
        temp = []
        for i in range(60000):
            for t in range(255):  # or it can be 256
                input_spike_state, input_img_label = input_layer.forward(i, t)
                if np.sum(input_spike_state) != 0:
                    output_spike_state, valid_train = hidden_1.forward(input_spike_state, input_img_label, t)
                    output_spike_state, _ = hidden_2.forward(output_spike_state, input_img_label, t)
                    output_spike_state, _ = output_layer.forward(output_spike_state, input_img_label, t)

            error = output_layer.calc_error()
            error = hidden_2.calc_error(error)
            error = hidden_1.calc_error(error)

            output_layer.backward(lr)
            hidden_2.backward(lr)
            hidden_1.backward(lr)

            predict_ans = output_layer.class_result()
            temp.append(predict_ans)

            hidden_1.clear()
            hidden_2.clear()
            output_layer.clear()

        train_acc.append(temp)
        end_train = time.time()
        train_acc_ite = np.array(train_acc[ite])
        right = len(np.where(train_acc_ite == 1)[0])  # 这玩意可能有问题！！！
        wrong = len(np.where(train_acc_ite == 0)[0])
        unkown = len(np.where(train_acc_ite == -1)[0])

        print('train acc ratio is: ', right / np.shape(train_acc_ite)[0], '& unkown ratio: ', unkown / np.shape(train_acc_ite)[0], '\n', '->train time is: ', (end_train - start_train) / 60, 'mins\n\n')

        # test:
        start_test = time.time()  # 测试计时
        temp = []
        for i in range(10000):
            for t in range(255):
                input_spike_state, input_img_label = input_layer.forward(i, t, mode='test')
                if np.sum(input_spike_state) != 0:
                    output_spike_state, valid_train = hidden_1.forward(input_spike_state, input_img_label, t)
                    output_spike_state, _ = hidden_2.forward(output_spike_state, input_img_label, t)
                    output_spike_state, _ = output_layer.forward(output_spike_state, input_img_label, t)

            predict_ans = output_layer.class_result()
            temp.append(predict_ans)

            hidden_1.clear()
            hidden_2.clear()
            output_layer.clear()

        test_acc.append(temp)
        end_test = time.time()

        test_acc_ite = np.array(test_acc[ite])
        right = len(np.where(test_acc_ite == 1)[0])
        wrong = len(np.where(test_acc_ite == 0)[0])
        unkown = len(np.where(test_acc_ite == -1)[0])
        print('test acc ratio is: ', right / np.shape(test_acc_ite)[0], '& unkown ratio: ', unkown / np.shape(test_acc_ite)[0], '\n', '->test time is: ', (end_test - start_test) / 60, 'mins\n\n')
