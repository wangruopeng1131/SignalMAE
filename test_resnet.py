import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from joblib import delayed, Parallel
import itertools
from dataIO.load_dataset import NoiseDataset
from colossalai.utils import get_dataloader

root = r"/home/wangruopeng/noise_16s/"
channels = ['I', 'II', 'V1', 'V5']
labels = ['A', 'B', 'C', 'D']
num_files = [i for i in range(10)]
file_names = ['data', 'beat', 'noise']
num = [i for i in range(10)]
save_root = r'/home/wangruopeng/database/noise_data'
Lmap = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
file_path = []
for r, c, l, n in itertools.product([root], channels, labels, num):
    file_path.append((os.path.join(r, c, l), n))


def preprocess(path_num):
    def _get_labels(label: int, noise: list, lv: list = None):
        l = np.zeros(4000)
        times = np.arange(4000)
        if not lv:
            for i in range(0, len(noise), 2):
                left = noise[i]
                right = noise[i + 1]
                where = np.where(((times > left) & (times < right)))
                l[where] = label
        else:
            for i in range(0, len(lv), 2):
                left = lv[i]
                right = lv[i + 1]
                where = np.where(((times > left) & (times < right)))
                l[where] = label

        labels = np.zeros(16)
        for i in range(16):
            split = l[i * 250: (i + 1) * 250]
            count = dict(Counter(split))
            if 0 in count.keys():
                count_0 = count[0]
                if count_0 < 250 * 0.8:
                    labels[i] = label
            else:
                labels[i] = label

        return labels

    path, num = path_num
    data_path = os.path.join(path, "data_{}.DAT".format(num))
    beat_path = os.path.join(path, "beat_{}.DAT".format(num))
    noise_path = None
    # lv_path = None
    label = path[-1]
    if label in ['B', 'C', 'D', 'E']:
        noise_path = os.path.join(path, "noise_{}.DAT".format(num))
    # if label == 'E':
    #     lv_path = os.path.join(path, "lv_{}.DAT".format(num))

    data_b = open(data_path, 'rb')
    beat_b = open(beat_path, 'rb')
    if noise_path:
        noise_b = open(noise_path, 'rb')
    # if lv_path:
    #     lv_b = open(lv_path, 'rb')
    length = os.stat(data_path).st_size // 16004
    for i in range(length):
        dic = {'data': None, 'beat_length': None, 'beat_label': None, 'peak': None, 'noise_II_I': None,
               'noise_II_II': None,
               'noise_III': None, 'lv_length': None, 'label': label}

        # data
        data_b.read(4)  # efid
        dd = struct.unpack('f' * 4000, data_b.read(16000))  # data
        dd = np.array(dd)
        dic['data'] = dd

        # beat
        beat_b.read(4)  # efid
        beat_length = struct.unpack('B', beat_b.read(1))[0]  # 16s片段中的心搏数量
        dic['beat_length'] = beat_length

        # 80个心搏分类标签
        beat_label = struct.unpack('b' * 80, beat_b.read(80))
        beat_label = np.array(beat_label)
        beat_label = beat_label[beat_label != -1]
        dic['beat_label'] = beat_label

        # 80个心搏R的相对位置点
        peak = struct.unpack('h' * 80, beat_b.read(160))
        peak = np.array(peak)
        peak = peak[peak != -1]
        dic['peak'] = peak

        labels = np.zeros(16)
        if noise_path:
            # noise
            noise_b.read(4)
            num_noise_II_I = struct.unpack('B', noise_b.read(1))[0]
            num_noise_II_II = struct.unpack('B', noise_b.read(1))[0]
            num_noise_III = struct.unpack('B', noise_b.read(1))[0]

            noise_II_I = np.array(struct.unpack('h' * 20, noise_b.read(40)))
            noise_II_II = np.array(struct.unpack('h' * 20, noise_b.read(40)))
            noise_III = np.array(struct.unpack('h' * 20, noise_b.read(40)))
            noise_II_I = noise_II_I[noise_II_I != -1]
            noise_II_II = noise_II_II[noise_II_II != -1]
            noise_III = noise_III[noise_III != -1]
            noise_II_I = None if len(noise_II_I) == 0 else noise_II_I
            noise_II_II = None if len(noise_II_II) == 0 else noise_II_II
            noise_III = None if len(noise_III) == 0 else noise_III
            noise = []
            if noise_II_I is not None:
                noise += list(noise_II_I)
            elif noise_II_II is not None:
                noise += list(noise_II_II)
            elif noise_III is not None:
                noise += list(noise_III)
            if noise:
                labels = _get_labels(Lmap[label], noise)

        # if lv_path:
        #     lv_b.read(4)
        #     lv_length = struct.unpack('B', lv_b.read(1))[0]
        #
        #     lv = struct.unpack('h' * 20, lv_b.read(40))
        #     lv = np.array(lv)
        #     lv = lv[lv != -1]
        #     lv = list(lv)
        #     labels = _get_labels(Lmap[label], noise, lv)

        dic['labels'] = labels
        path_list = path.split('/')

        save_path = os.path.join(save_root, f'{path_list[-2]}_{path_list[-1]}_data{num}_{i}.npy')
        print(save_path)
        np.save(save_path, dic)


if __name__ == '__main__':
    # Parallel(n_jobs=30)(delayed(preprocess)(path) for path in file_path)
    dataset = NoiseDataset(train=True, NOISE_DATA_ROOT=r'/home/wangruopeng/database/noise_data')
    train_dataloader = get_dataloader(
        dataset=dataset,
        shuffle=True,
        batch_size=10,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    for i, (d, l) in enumerate(train_dataloader):
        if 4 in l:
            print(1)

