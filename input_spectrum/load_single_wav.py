import numpy as np
import torch
import matplotlib.pyplot as plt
from models.utils import weights_init
import torch.utils.data
from datasets import *
import datasets.utils as U

import os
import wavio
import models
import opts

opt = opts.parse()

def load_wav(wav_file):

    sound = wavio.read(wav_file).data.T[0]
    # sound, fs = sf.read(wav_file)
    start = sound.nonzero()[0].min()  # 获取 sound 数据不为 0 的索引最小值
    end = sound.nonzero()[0].max()  # 获取最大值
    sound = sound[start: end + 1]  # Remove silent sections
    label = int(os.path.splitext(wav_file)[0].split('-')[-1])
    return sound, label


def preprocess(sound):
    funcs = [U.padding(opt.inputLength // 2),
             U.normalize(32768.0),
             U.multi_crop(opt.inputLength, opt.nCrops)
             ]

    for f in funcs:
        sound = f(sound)

    return sound.astype(np.float32)

def load_model(model_path):

    # model = models.EnvNet2(50)
    model = models.EnvNet3(50)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    return model


if __name__ == '__main__':
    model_path = 'C:/Users/zcf/Desktop/model_2000_1.bin'
    model = load_model(model_path)
    model.eval()

    src_path = 'C:/Users/zcf/Desktop/envnet3/figure1/1-30709-A-23.wav'
    sound, label = load_wav(src_path)

    # print(f'sound = {sound}')
    # print(f'sound.max() = {sound.max()}')
    # print(f'len(sound) = {len(sound)}')

    sound = preprocess(sound)
    # print(f'sound.shape = {sound.shape}')

    datas = torch.utils.data.DataLoader(sound, batch_size=10)

    with torch.no_grad():
        for data in datas:
            input = data[:, None, None, :]
            # print(f'input.shape = {input.shape}')
            output, y4, y5 = model(input)

    print(f'output.shape = {output.shape}')
    print(f'y4.shape = {y4.shape}')
    print(f'y5.shape = {y5.shape}')
    # print(f'y5.shape = {y5.shape}')



    # print(f'data.shape = {data.shape}')
    # data = data.data.numpy()
    # print(f'data.shape = {data.shape}')
    # plt.imshow(data)
    # plt.show()


    # print(f'output.shape = {output.shape}')
    # output = output.data.numpy()
    # plt.imshow(output)
    # plt.savefig('C:/Users/zcf/Desktop/envnet3/figure4/out_plot.png')
    # plt.show()


    # print(f'y.shape = {y.shape}')
    # y = y.permute(0,2,3,1)
    # print(f'y.shape = {y.shape}')


    # y = y.data.numpy()
    # y = np.mean(y, axis=1)
    # print(f'y.shape = {y.shape}')
    # print(f'y.len = {len(y)}')
    # for i in range(len(y)):
    #     plt.imshow(y[i])
    #     plt.savefig('C:/Users/zcf/Desktop//envnet3/figure1/dense_block1/{}.png'.format(i))

    y4 = y4.data.numpy()
    # y1 = np.mean(y1, axis=1)
    # print(f'y0.shape = {y0.shape}')
    # print(f'y0.len = {len(y0)}')
    for i in range(10):
        plt.imshow(y4[0][i], cmap="Blues_r")
        plt.savefig('C:/Users/zcf/Desktop//envnet3/figure1/dense_block3/{}.png'.format(i))

    y5 = y5.data.numpy()
    # y1 = np.mean(y1, axis=1)
    # print(f'y1.shape = {y1.shape}')
    # print(f'y1.len = {len(y1)}')
    for i in range(10):
        plt.imshow(y5[0][i], cmap="Blues_r")
        plt.savefig('C:/Users/zcf/Desktop//envnet3/figure1/transition_block3/{}.png'.format(i))


    # y = np.mean(y, axis=0)
    # print(f'y.shape = {y.shape}')i
    # plt.imshow(y)
    # plt.show()










