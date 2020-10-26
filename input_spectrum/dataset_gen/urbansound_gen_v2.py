"""
 Dataset preparation code for UrbanSound8k [Salamon, 2014].
 Usage: python urbansound_gen.py [path]
 Original dataset should be downloaded in [path]/urbansound8k/.
 FFmpeg should be installed.

"""

import sys
import os
import subprocess
import argparse

import glob
import random
import numpy as np
import wavio
import librosa
from collections import Counter


fs = 44100

def main(data_path):
    # data_path = os.path.join(sys.argv[1], 'urbansound8k')
    # fs_list = [16000, 44100]
    #
    # # Convert sampling rate
    # for fs in fs_list:
    #     convert_fs(os.path.join(data_path, 'UrbanSound8K/audio'),
    #                os.path.join(data_path, 'wav{}'.format(fs // 1000)),
    #                fs)

    # Create npz files
    src_path = data_path

    dst_path = os.path.join(data_path, 'wav{}_v3'.format(fs // 1000))

    create_dataset(src_path, dst_path + '.npz')


def convert_fs(src_path, dst_path, fs):
    print('* {} -> {}'.format(src_path, dst_path))
    os.mkdir(dst_path)
    for fold in sorted(os.listdir(src_path)):
        if os.path.isdir(os.path.join(src_path, fold)):
            os.mkdir(os.path.join(dst_path, fold))
            for src_file in sorted(glob.glob(os.path.join(src_path, fold, '*.wav'))):
                dst_file = src_file.replace(src_path, dst_path)
                subprocess.call('ffmpeg -i {} -ac 1 -ar {} -acodec pcm_s16le -loglevel error -y {}'.format(
                    src_file, fs, dst_file), shell=True)


def create_dataset(src_path, dst_path):
    print('* {} -> {}'.format(src_path, dst_path))
    dataset = {}

    dataset['train'] = {}
    dataset['test'] = {}

    train_sounds = []
    train_labels = []
    test_sounds = []
    test_labels = []
    all_wav = []

    for fold in range(1, 11):
        all_wav += glob.glob(os.path.join(src_path, 'fold{}'.format(fold), '*.wav'))

    random.seed(123)
    random.shuffle(all_wav)

    for wav_file in all_wav:
        # sound = wavio.read(wav_file).data.T[0]
        sound, _ = librosa.load(wav_file, sr=44100)
        start = sound.nonzero()[0].min()
        end = sound.nonzero()[0].max()
        sound = sound[start: end + 1]  # Remove silent sections
        label = int(wav_file.split('/')[-1].split('-')[1])

        temp = random.randint(1,10)

        if temp < 3:
            # print(temp)
            # print(wav_file)
            test_sounds.append(sound)
            test_labels.append(label)

        else:
            train_sounds.append(sound)
            train_labels.append(label)


    dataset['train']['sounds'] = train_sounds
    dataset['train']['labels'] = train_labels
    dataset['test']['sounds'] = test_sounds
    dataset['test']['labels'] = test_labels



    np.savez(dst_path, **dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process Urbansound8k')
    parser.add_argument('--path', type=str, default='/home/yons/chengfei/urbansound_train_v2/datasets/UrbanSound8K',
                        help='urbansound8k path')
    # parser.add_argument('--path', type=str, default='D:/ESC实验记录/datasets/UrbanSound8K',
    #                     help='urbansound8k path')
    args = parser.parse_args()
    main(args.path)
