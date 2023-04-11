import os
from scipy import io
from scipy.signal import resample_poly
import numpy as np
from joblib import delayed, Parallel


def pipline(file_path: str):
    save_root = r'/home/wangruopeng/pretrain-data/prepared_data'
    save_names = os.path.split(file_path)[-1].split('.')[0]

    mat = io.loadmat(path_list[0])['val'].astype(np.float64)
    mat = resample_poly(mat, 250, 500, axis=-1)
    x, y = mat.shape
    print(save_names)
    for i in range(x):
        save_name = '{}-{}.npy'.format(save_names, i)
        save_name = os.path.join(save_root, save_name)
        np.save(save_name, mat[i])


if __name__ == "__main__":
    path = r'/home/wangruopeng/pretrain-data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords'
    path_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'mat' in file:
                path_list.append(os.path.join(root, file))
    Parallel(n_jobs=20)(delayed(pipline)(path) for path in path_list)
