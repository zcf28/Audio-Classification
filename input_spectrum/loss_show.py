import numpy as np
import matplotlib.pyplot as plt
import re


def read_data(data_path):
    data = open(data_path).readlines()

    Train_loss = []
    Val_loss = []
    Accuracy = []
    for i in range(len(data)):
        if data[i][0] != 'E':
            continue
        data[i] = data[i].strip()
        nums = re.findall(r"\d+\.?\d*", data[i])


        Train_loss.append(float(nums[-3]))
        Val_loss.append(float(nums[-2]))
        Accuracy.append(float(nums[-1]))

    return Train_loss, Val_loss, Accuracy

if __name__ == '__main__':
    data1_path = 'C:/Users/zcf/Desktop/1.txt'
    Train_loss1, Val_loss1, Accuracy1 = read_data(data1_path)

    # data2_path = 'C:/Users/zcf/Desktop/2.txt'
    # Train_loss2, Val_loss2, Accuracy2 = read_data(data2_path)
    #
    # data3_path = 'C:/Users/zcf/Desktop/3.txt'
    # Train_loss3, Val_loss3, Accuracy3 = read_data(data3_path)

    Epoch = np.arange(0, len(Accuracy1), 1)

    # print(Train_loss1[:100])
    # print(Train_loss2[:100])
    # plt.figure(figsize=(16, 9))
    # plt.xlim(0, 2000, 1)
    # plt.ylim(0, 0.1)

    # Accuracy1 = Accuracy1[1000:]
    print(f'len(Accuracy1) = {len(Accuracy1)}')
    print(f'np.max(Accuracy1) = {np.max(Accuracy1)}')
    # print(f'len(Accuracy2) = {len(Accuracy2)}')
    # print(f'np.max(Accuracy2) = {np.max(Accuracy2)}')
    # print(f'len(Accuracy3) = {len(Accuracy3)}')
    # print(f'np.max(Accuracy3) = {np.max(Accuracy3)}')


    # plt.plot(Epoch, np.array(Accuracy1), Epoch, np.array(Accuracy2), Epoch, np.array(Accuracy3))
    # plt.plot(Epoch, np.array(Train_loss1), Epoch, np.array(Val_loss1))
    plt.plot(Epoch, np.array(Accuracy1))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.legend(('1', '2', '3'))
    # plt.title("envnet3_4")

    plt.show()

