import numpy as np
import matplotlib.pyplot as plt
import wavio
import librosa
import librosa.display
import torch
import random
import seaborn as sns
import pandas as pd

from python_speech_features.base import delta
from spafe.features import mfcc, gfcc
from datasets.utils import spectrum_standardization

def load_wav(wav_path):

    # sound, _ = librosa.load(wav_path, sr=44100)
    sound = wavio.read(wav_path).data.T[0]

    start = np.nonzero(sound)[0][0]
    end = np.nonzero(sound)[0][-1]
    print(f'start = {start}')
    print(f'end = {end}')
    return sound[start:end+1]

def random_crop(sound):

    sound = np.pad(sound, 66650//2, 'constant')

    stride = (len(sound) - 66650) // 9
    sounds = []
    for i in range(10):
        sub_sound = sound[stride * i: stride * i + 66650]
        sounds.append(sub_sound)
    return np.array(sounds)

def get_spec(sound):
    mel = librosa.feature.melspectrogram(sound, sr=44100, n_fft=888, hop_length=445, n_mels=128)
    log_mel = librosa.power_to_db(mel).T
    return log_mel


if __name__ == '__main__':
    wav_path = "C:/Users/zcf/Desktop/1.wav"
    sound = load_wav(wav_path)
    sounds = random_crop(sound)
    # print(f'sounds = {sounds}')

    datas = []
    for i in range(10):

        log_mel = get_spec(sounds[i]/32768.0)
        log_mel = spectrum_standardization(log_mel, -40.84, 23.92)
        data = np.concatenate(log_mel, axis=0)
        datas.append(data)

    # print(datas)

    data = pd.DataFrame({"0": datas[0], "1": datas[1], "2": datas[2], "3": datas[3],
                         "4": datas[4],"5": datas[5],"6": datas[6],"7": datas[7],
                         "8": datas[8],"9": datas[9],})
    data.boxplot()
    plt.title('gt of spafe')
    plt.show()