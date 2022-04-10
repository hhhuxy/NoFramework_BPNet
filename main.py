# This is a task of recognizing handwriting digits
# author: hxy
# TASK 1 for Course design of machine intelligence
import matplotlib.pyplot as plt
import bp_model
import pickle
import gzip
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Hand writing digits recognition')

parser.add_argument('-s', '--batch_size', type=int, nargs='?', default=32)
parser.add_argument('-l', '--layer_sizes', nargs='*', type=int, default=[784, 100, 10])
parser.add_argument('-e', '--epoch', nargs='?', default=20, type=int)
parser.add_argument('-q', '--quiet', nargs='?', default=False, type=bool)
parser.add_argument('-p', '--print_score', type=bool, nargs='?', default=True)
parser.add_argument('-o', '--optimizer', type=str, nargs='?', default='minibatch')
parser.add_argument('-f', '--activation_func', type=str, nargs='?', default='sigmoid')
parser.add_argument('-i', '--init_method', type=str, nargs='?', default='uniform')
parser.add_argument('--init_bound', type=float, nargs='*', default=[-0.5, 0.5])

args = parser.parse_args()


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    vec_training_results = [convert2one_hot_vec(y) for y in tr_d[1]]
    one_hot_training_data = zip(tr_d[0], vec_training_results)
    validation_data = zip(va_d[0], va_d[1])
    test_data = zip(te_d[0], te_d[1])
    return one_hot_training_data, validation_data, test_data


def convert2one_hot_vec(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e


# 绘制单个图像
def draw(data, width, height):
    """
    :param data: 一张图片的灰度值数据，形状：1x(width, height)向量(array)
    :param width: 图片宽度
    :param height: 图片高度
    :return: None
    """
    d = data.reshape(width, height)
    fig = plt.figure()
    plt.imshow(d, cmap='gray')
    plt.show()


# main
if __name__ == '__main__':
    one_hot_training_data, validation_data, test_data = load_data_wrapper()
    # draw(train_data[1][0], 28, 28)
    model = bp_model.Model(layer_sizes=args.layer_sizes, weight_init_method=args.init_method, init_bound=args.init_bound,
                           activation_func_name=args.activation_func)
    print(args)
    assert args.optimizer in ['minibatch', 'standard']
    if args.optimizer is 'minibatch':
        f1_train, f1_valid = model.minibatch_sgd(one_hot_training_data, validation_data,  print_score=bool(args.print_score),
                                                 epoch=args.epoch, minibatch_size=args.batch_size, step=0.1)
    elif args.optimizer is 'standard':
        f1_train, f1_valid = model.standard_sgd(one_hot_training_data, validation_data, print_score=bool(args.print_score),
                                                epoch=args.epoch, step=0.08)
    plt.figure()
    x = range(args.epoch)
    plt.plot(x, f1_train, label='train dataset')
    plt.plot(x, f1_valid, label='validation dataset')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('F1 score')
    plt.legend()
    plt.show()
    test_data = list(test_data)
    print('test data:')
    model.print_score(test_data)

