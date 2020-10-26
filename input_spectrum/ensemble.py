import torch
import os
import opts
from datasets import *
import models
import torch.utils.data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

opt = opts.parse()
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print("{} is used.".format(device))

def load_model(model_path1:str, model_path2:str):
    model1 = models.EnvNet5(50).cuda()
    model2 = models.EnvNet6(50).cuda()

    model1.load_state_dict(torch.load(model_path1, map_location=lambda storage, loc: storage))
    model2.load_state_dict(torch.load(model_path2, map_location=lambda storage, loc: storage))

    model1.eval()
    model2.eval()

    return model1, model2

def data_test(models):

    test_fold = 5
    print("Data Get fold {}".format(test_fold))

    _, validation_generator = get_data_generators(opt, test_fold)

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

            outputs = models[0](inputs)
            outputs += models[1](inputs)
            print(outputs.size())



            outputs_total = outputs.data.reshape(outputs.shape[0] // opt.nCrops, opt.nCrops, outputs.shape[1])

            outputs_total = torch.mean(outputs_total, 1)


            _, answer = torch.max(labels, 1)
            _, predicted = torch.max(outputs_total, 1)
            total += labels.size(0)
            # print(total)
            correct += (predicted == answer).sum().item()

            # answer = answer.data.cpu().numpy()
            # predicted = predicted.data.cpu().numpy()
            # X_true += [i for i in answer]
            # Y_predict += [i for i in predicted]



    accuracy = 100 * correct / total

    print(f'accuracy = {accuracy}')
    # print(f'X_true = {X_true}')
    # print(f'Y_predict = {Y_predict}')

    # return accuracy, X_true, Y_predict



if __name__ == '__main__':

    model_path1 = 'D:/ESC实验记录/model/input_spectrum/result/esc50/model/envnet5_1_2/fold1-5/model_1000_5.bin'
    model_path2 = 'D:/ESC实验记录/model/input_spectrum/result/esc50/model/envnet6_1_1/fold1-5/model_1000_5.bin'

    data_path = 'D:/ESC实验记录/datasets/esc50/wav44.npz'

    model1, model2 = load_model(model_path1, model_path2)

    models = [model1, model2]
    data_test(models)