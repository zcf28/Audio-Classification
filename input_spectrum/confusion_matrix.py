import torch
import os
import opts
from datasets import *
import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

opt = opts.parse()
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print("{} is used.".format(device))


def crossentropy_loss(input, target):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    crossEntropy = - torch.sum(target * logsoftmax(input))
    return crossEntropy / input.shape[0]

def data_test(model_path):

    test_fold = 5
    print("Data Get fold {}".format(test_fold))

    model = models.EnvNet6(opt.nClasses).to(device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()

    _, validation_generator = get_data_generators(opt, test_fold)

    eval_loss = 0.0
    correct = 0.0
    total = 0.0

    X_true = []
    Y_predict = []

    with torch.no_grad():
        for data in validation_generator:
            # print(data) # 包含数据和标签
            input_array, label_array = data[0], data[1]

            input_array = input_array.reshape(input_array.shape[0] * opt.nCrops, input_array.shape[2],
                                              input_array.shape[3])

            inputs = input_array[:, None, :, :].to(device)
            labels = label_array.to(device)

            outputs = model(inputs)


            outputs_total = outputs.data.reshape(outputs.shape[0] // opt.nCrops, opt.nCrops, outputs.shape[1])

            outputs_total = torch.mean(outputs_total, 1)


            _, answer = torch.max(labels, 1)
            _, predicted = torch.max(outputs_total, 1)
            total += labels.size(0)
            # print(total)
            correct += (predicted == answer).sum().item()

            answer = answer.data.cpu().numpy()
            predicted = predicted.data.cpu().numpy()
            X_true += [i for i in answer]
            Y_predict += [i for i in predicted]

            loss = crossentropy_loss(outputs_total.data, labels.data)

            eval_loss += loss.item()


    eval_loss = eval_loss/len(validation_generator)
    accuracy = 100 * correct / total
    print(f'eval_loss = {eval_loss}')
    print(f'accuracy = {accuracy}')
    print(f'X_true = {X_true}')
    print(f'Y_predict = {Y_predict}')

    return eval_loss, accuracy, X_true, Y_predict


def show_confusion_matrix(cm, savename = None, normalize=False, title='Confusion Matrix Proposed4', cmap=plt.cm.Blues):

    dicts = {17: '倒水', 18: '抽水马桶', 20: '哭闹的宝宝', 21: '打喷嚏', 23: '呼吸', 24: '咳嗽', 25: '脚步声', 26: '笑声', \
             27: '刷牙', 28: '打鼾', 29: '喝水', 36: '真空吸尘器', 37: '闹钟', 38: '时钟滴答声', 39: '玻璃破碎'}

    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['font.size'] = '15'  # 设置字体大小

    # classes表示不同类别的名称，比如这有10个类别

    classes = [str(dicts[i]) for i in [17,18,20,21,23,24,25,26,27,28,29,36,37,38,39]]
    # classes = [str(i) for i in range(10)]
    # '0/dog', '1/rain', '2/sea_waves', '3/crying_baby', '4/clock_tick', '5/sneezing'，
    # ‘6/helicopter’， ‘7/chainsaw’， ‘8/rooster’， ‘9/crackling_fire’

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)
    plt.figure(figsize=(12,11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.ylabel('真实事件', c='r')
    plt.xlabel('预测事件', c='r')

    if savename:
        plt.savefig(savename, format='png')

    plt.show()


if __name__ == '__main__':

    # model_path = '/home/yons/chengfei/spectrum_learning/result/esc50/envnet6_1_1/model_1000_5.bin'
    # _, _, y_true, y_pred = data_test(model_path)
    # print(f'y_true = {y_true}')
    # print(f'y_pred = {y_pred}')


    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # np.save("/home/yons/chengfei/test/result/fold5", cm)

    # cm = np.load('/home/yons/chengfei/test/result/fold1.npy')
    # for i in range(2, 6):
    #     cm += np.load(f'/home/yons/chengfei/test/result/fold{i}.npy')
    #
    # np.save("/home/yons/chengfei/test/result/mean", cm//5)
    cm = np.load("C:/Users/zcf/Desktop/mean.npy",)
    print(cm.shape)
    dicts = {17:'倒水', 18:'抽水马桶', 20:'哭闹的宝宝', 21:'打喷嚏', 23:'呼吸', 24:'咳嗽', 25:'脚步声', 26:'笑声',\
            27:'刷牙', 28:'打鼾', 29:'喝水', 36:'真空吸尘器', 37:'闹钟', 38:'时钟滴答声', 39:'玻璃破碎'}
    nums = [17,18,20,21,23,24,25,26,27,28,29,36,37,38,39]
    ls = []
    for i in nums:
        temp = []
        for j in nums:
            temp.append(cm[i][j])
        ls.append(temp)

    print(np.array(ls))
    show_confusion_matrix(np.array(ls), savename='confusion_matrix.png')





