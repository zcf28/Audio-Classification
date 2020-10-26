import os
import glob
import math
from pydub import AudioSegment
import random



def mix_audio(src_path, tar_path):
    random.seed(123)

    if os.path.exists(tar_path) == False:
        os.mkdir(tar_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        print(f'src_file = {src_file}')
        tar_file = src_file.replace(src_path, tar_path)
        print(f'tar_file = {tar_file}')
        source = AudioSegment.from_wav(src_file)
        s_dB = 20 * math.log10(source.rms)
        m = random.randint(1, 5)
        background =AudioSegment.from_wav('/home/yons/chengfei/data/music/{}.wav'.format(m))
        # background = AudioSegment.from_wav('/home/yons/chengfei/data/background music/noise/1.wav')
        b_dB = 20 * math.log10(background.rms)

        if s_dB > b_dB:
            if s_dB - b_dB >= 10:
                background = background.apply_gain(+(s_dB - 10 - b_dB))
            else:
                background = background.apply_gain(-(b_dB - (s_dB - 10)))
        else:
            background = background.apply_gain(-(b_dB - (s_dB - 10)))

        print(f's_dB = {20 * math.log10(source.rms)}')
        print(f'd_dB = {20 * math.log10(background.rms)}')
        combined = source.overlay(background)
        combined.export(tar_file, format="wav")

if __name__ == '__main__':
    src_path = '/home/yons/chengfei/data/audio/'
    # src_path = 'D:/data/audio'
    tar_path = '/home/yons/chengfei/data/mix-music-drop10dB/'
    mix_audio(src_path, tar_path)