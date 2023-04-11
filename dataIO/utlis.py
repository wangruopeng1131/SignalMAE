import os
import struct
from collections import Counter
import numpy as np
from joblib import delayed, Parallel


def get_eeg_path(root):
    file_root = [os.path.join(root, i) for i in os.listdir(root) if 'chb' in i]
    files = []
    for f in file_root:
        for ff in os.listdir(f):
            if 'seizures' not in ff and 'txt' not in ff:
                files.append(os.path.join(f, ff))
    return files


def _get(path):
    files = []
    for file in os.listdir(path):
        f = file.split('.')[0]
        f = os.path.join(path, f)
        if f not in files and 'RECORDS' not in f:
            files.append(f)
    return files


def get_path(root):
    # root = r'/home/wangruopeng/pretrain-data/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0'
    files_root = [os.path.join(root, i) for i in os.listdir(root) if 'p' in i]
    files_path = []
    for files in files_root:
        for i in os.listdir(files):
            if 'RECORDS' not in i:
                files_path.append(os.path.join(files, i))

    tmp = Parallel(n_jobs=30)(delayed(_get)(path) for path in files_path)
    files_path = []
    for file in tmp:
        for i in file:
            files_path.append(i)

    return files_path


def preprocess(path_num):
    def _get_labels(label: int, noise: list, lv: list = None):
        times = np.zeros(4000)
        if not lv:
            for i in range(0, len(noise), 2):
                left = noise[i]
                right = noise[i + 1]
                where = np.where(((times > left) & (times < right)))
                times[where] = label
        else:
            for i in range(0, len(lv), 2):
                left = lv[i]
                right = lv[i + 1]
                where = np.where(((times > left) & (times < right)))
                times[where] = label

        labels = np.zeros(16)
        for i in range(16):
            split = times[i * 250: (i + 1) * 250]
            count = dict(Counter(split))
            count_0 = count[0]
            if count_0 < 250 * 0.8:
                labels[i] = label

        return labels

    save_root = r'~'
    path, num = path_num
    data_path = os.path.join(path, "data_{}.DAT".format(num))
    beat_path = os.path.join(path, "beat_{}.DAT".format(num))
    noise_path = None
    lv_path = None
    label = path[-1]
    if label in ['B', 'C', 'D', 'E']:
        noise_path = os.path.join(path, "noise_{}.DAT".format(num))
    if label == 'E':
        lv_path = os.path.join(path, "lv_{}.DAT".format(num))

    data_b = open(data_path, 'rb')
    beat_b = open(beat_path, 'rb')
    if noise_path:
        noise_b = open(noise_path, 'rb')
    if lv_path:
        lv_b = open(lv_path, 'rb')
    length = os.stat(data_path).st_size // 16004
    train = int(length * 0.7)
    for i in range(length):
        dic = {'data': None, 'beat_length': None, 'beat_label': None, 'peak': None, 'noise_II_I': None,
               'noise_II_II': None,
               'noise_III': None, 'lv': None, 'lv_length': None, 'label': label}

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
            if noise_II_I:
                noise += list(noise_II_I)
            elif noise_II_II:
                noise += list(noise_II_II)
            elif noise_III:
                noise += list(noise_III)
            labels = _get_labels(label, noise)

        if lv_path:
            lv_b.read(4)
            lv_length = struct.unpack('B', lv_b.read(1))[0]

            lv = struct.unpack('h' * 20, lv_b.read(40))
            lv = np.array(lv)
            lv = lv[lv != -1]
            lv = list(lv)
            labels = _get_labels(label, noise, lv)

        dic['labels'] = labels
        path_list = path.split('/')

        if i < train:
            save_path = os.path.join(save_root, 'train', f'{path_list[-2]}_{path_list[-1]}_data{num}_{i}.npy')
        else:
            save_path = os.path.join(save_root, 'test', f'{path_list[-2]}_{path_list[-1]}_data{num}_{i}.npy')
        print(save_path)
        np.save(save_path, dic)
