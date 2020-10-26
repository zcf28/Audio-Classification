import numpy as np
import random
import librosa
from spafe.features import gfcc
from python_speech_features.base import delta

# Default data augmentation
def padding(pad):
    def f(sound):

        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f


def normalize(factor):
    def f(sound):
        return sound / factor

    return f

def sp_normalization(sp):
    #sp.shape (T, D)
    sp_con = np.concatenate(sp, axis=0)
    sp_max = np.max(sp_con)
    sp_min = np.min(sp_con)
    return 2*(sp-sp_min)/(sp_max-sp_min+1e-8) - 1


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):

        stride = (len(sound) - input_length) // (n_crops - 1)
        # GT = []
        logMel = []
        for i in range(n_crops):
            sub_sound = sound[stride * i: stride * i + input_length]

            mel = librosa.feature.melspectrogram(sub_sound, sr=44100, n_fft=888, hop_length=445, n_mels=128)
            log_mel = librosa.power_to_db(mel).T
            log_mel = sp_normalization(log_mel)

            # delta1 = delta(log_mel, 2)
            # delta2 = delta(delta1, 2)
            # log_mel = sp_normalization(log_mel)
            # delta1 = sp_normalization(delta1)
            # delta2 = sp_normalization(delta2)
            # log_mels = []
            # log_mels.append(log_mel)
            # log_mels.append(delta1)
            # log_mels.append(delta2)
            #
            # _, gt, _ = gfcc.gfcc(sub_sound, fs=44100, win_len=0.02, win_hop=0.01, nfilts=128)
            # gt = sp_normalization(gt)
            # log_mels.append(log_mel)
            # log_mels.append(gt)

            # GT.append(gt)
            logMel.append(log_mel)

        # return np.array(GT)
        return np.array(logMel)

    return f


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound

# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line
