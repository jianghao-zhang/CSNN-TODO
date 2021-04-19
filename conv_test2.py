import time
import numpy as np
from data.mnist_dataset import load
from layers.coding import SNNInput
from layers.conv import SNNConv2d
from layers.linear import SNNLinear
from layers.pool import SNNMaxPooling

if __name__ == "__main__":

    # hyper parameters:
    network_architecture = [800, 100]
    vth = [5, 5, 15]
    lr1, lr2 = 1e-2, 1e-5
    beta1, beta2 = 3, 3
    tau1, tau2 = 1000, 1000
    k1 = 5
    k2_1, k2_2 = 0.1, 0.01

    # todo-todo-todo:
    # todo-todo-todo-todo: 线性的都过拟合了！！

    # 到底是学习率太大还是太小？反正是朝上一个的反向冲了，感觉是因为惯性冲太猛了！！，指下面这组参数：
    # 感觉上是矫枉过正
    # 那么调整哪些参数可以抑制这种过头的变化呢？
    # lr1 & lr2
    # k1
    # k2_1, k2_2
    # network_architecture = [800, 500, 100]
    # vth = [5, 5, 5, 15, 15]
    # lr1, lr2 = 1e-4, 1e-3
    # beta1, beta2 = 3, 3
    # tau1, tau2 = 200, 1000
    # k1 = 1e-4
    # k2_1, k2_2 = 0.1, 0.1

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
    conv_1 = SNNConv2d(in_channels=1, out_channels=16, final_size=100, input_size=28, next_size=1, kernal_size=5, stride=1, padding=0, k1=k1, k2=k2_1, tau=tau1, vth=vth[0], beta=beta1, lr=lr1, mode='dfa')
    pool_1 = SNNMaxPooling(shape=[24, 24, 16], kernal_size=2, stride=2)
    conv_2 = SNNConv2d(in_channels=16, out_channels=32, final_size=100, input_size=12, next_size=1, kernal_size=3, stride=1, padding=0, k1=k1, k2=k2_1, tau=tau1, vth=vth[1], beta=beta1, lr=lr1, mode='dfa')
    pool_2 = SNNMaxPooling(shape=[10, 10, 32], kernal_size=2, stride=2)
    # hidden_1 = SNNLinear(network_architecture[0], network_architecture[1], network_architecture[-1], k2_2, tau2, vth[2], beta2, lr2)
    output_layer = SNNLinear(network_architecture[0], network_architecture[-1], network_architecture[-1], k2_2, tau2, vth[1], beta2, lr2, class_num=10)

    # spike encoding:
    train_spike, validation_spike, test_spike = input_layer.temporal_encoding()

    # train:
    start_train = time.time()  # 训练计时
    for i in range(2000):
        for t in range(255):  # or it can be 256
            input_spike_state, input_img_label = input_layer.forward(i, t)
            if np.sum(input_spike_state) != 0:
                input_spike_state = np.reshape(input_spike_state, (28, 28, 1))
                NoneInput, output_spike_state = conv_1.forward(input_spike_state, t)
                if NoneInput is False:
                    output_spike_state = pool_1.forward(output_spike_state)
                    NoneInput, output_spike_state = conv_2.forward(output_spike_state, t)
                    if NoneInput is False:
                        output_spike_state = pool_2.forward(output_spike_state)
                        reshape_to_flatten = output_spike_state.shape
                        out_flatten = output_spike_state.reshape(-1)

                        # todo-todo-todo: out-flatten
                        # print('out_flatten is: ', np.sum(out_flatten))

                        # output_spike_state, valid_train = hidden_1.forward(out_flatten, t)
                        # output_spike_state, _ = hidden_2.forward(output_spike_state, t)
                        output_spike_state, _ = output_layer.forward(out_flatten, t)

        output_error = output_layer.calc_error(label=input_img_label)
        # hidden_2.calc_error(output_error)
        # hidden_1.calc_error(output_error)
        conv_2.calc_error(output_error)
        conv_1.calc_error(output_error)

        output_layer.backward(lr1)
        # hidden_2.backward(lr1)
        # hidden_1.backward(lr1)
        conv_2.backward(lr2)
        conv_1.backward(lr2)

        predict_ans = output_layer.class_result()
        train_acc.append(predict_ans)

        # hidden_1.clear()
        # hidden_2.clear()
        output_layer.clear()
        conv_1.clear()
        conv_2.clear()

    end_train = time.time()
    train_acc = np.array(train_acc)
    right = len(np.where(train_acc == 1)[0])  # 这玩意可能有问题！！！
    wrong = len(np.where(train_acc == 0)[0])
    unkown = len(np.where(train_acc == -1)[0])

    print('train acc ratio is: ', right / np.shape(train_acc)[0], '& unkown ratio: ', unkown / np.shape(train_acc)[0], '\n', '->train time is: ', (end_train - start_train) / 60, 'mins')

    # test:
    start_test = time.time()  # 测试计时
    for i in range(500):
        for t in range(255):
            input_spike_state, input_img_label = input_layer.forward(i, t, mode='test')
            if np.sum(input_spike_state) != 0:
                input_spike_state = np.reshape(input_spike_state, (28, 28, 1))
                NoneInput, output_spike_state = conv_1.forward(input_spike_state, t)
                if NoneInput is False:
                    output_spike_state = pool_1.forward(output_spike_state)
                    NoneInput, output_spike_state = conv_2.forward(output_spike_state, t)
                    if NoneInput is False:
                        output_spike_state = pool_2.forward(output_spike_state)

                        reshape_to_flatten = output_spike_state.shape
                        out_flatten = output_spike_state.reshape(-1)

                        # output_spike_state, valid_train = hidden_1.forward(out_flatten, input_img_label, t)
                        # output_spike_state, _ = hidden_2.forward(output_spike_state, input_img_label, t)
                        output_spike_state, _ = output_layer.forward(out_flatten, t)

        predict_ans = output_layer.class_result()
        test_acc.append(predict_ans)

        # hidden_1.clear()
        # hidden_2.clear()
        output_layer.clear()
        conv_2.clear()
        conv_1.clear()

    end_test = time.time()

    test_acc = np.array(test_acc)
    right = len(np.where(test_acc == 1)[0])
    wrong = len(np.where(test_acc == 0)[0])
    unkown = len(np.where(test_acc == -1)[0])
    print('test acc ratio is: ', right / np.shape(test_acc)[0], '& unkown ratio: ', unkown / np.shape(test_acc)[0], '\n', '->test time is: ', (end_test - start_test) / 60, 'mins')
