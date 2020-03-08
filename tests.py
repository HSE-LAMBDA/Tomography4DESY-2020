import numpy as np
from collections.abc import Iterable
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_slices(sino, angles, shifts=0, n_turns=8):
    '''
    Prepare slices of sinogram for testing Vo's method
     :param sino: prepared sinogram
     :param angles: angles of taken sinogram
     :param shifts: int or array of shifts of angles from halfturn
     :param n_turns: number of rotations of sinogram
     
     :return: prepared sino slices
    '''
    if not isinstance(shifts, Iterable):
        shifts = [shifts] * n_turns
    sino_slices = []
    for i in range(n_turns):
        if shifts[i] > 0 and i == n_turns - 1: continue  # skip not full sized 
        if shifts[i] < 0 and i == 0: continue             # skip not full sized 
        mask = ((i * np.pi + shifts[i] < angles.squeeze()) & (angles.squeeze() < (i + 1) * np.pi + shifts[i]))
        sino_slices.append(sino[mask])
    min_len = min(len(slice_) for slice_ in sino_slices)
    sino_slices = np.array([sino_slice[:min_len] for sino_slice in sino_slices]).transpose(1, 0, 2)
    return sino_slices


def test_rotations(func, sino_corrected, angles, n_experiments = 12, n_turns=8, *args, **kwargs):
    '''
    '''
    vo_centers = []
    for i in tqdm(range(n_experiments)):
        shifts = np.random.uniform(-np.pi * 1 / 2, np.pi * 1 / 2, size=n_turns)
        sino_slices = prepare_slices(sino_corrected, angles, shifts, n_turns=n_turns)
        vo_centers.append(func(sino_slices, *args, **kwargs))
    vo_centers = np.array(vo_centers)
    print('\nmean:', np.mean(sino_corrected.shape[1] / 2 - vo_centers), 'std:', np.std(sino_corrected.shape[1] / 2 - vo_centers))
    plt.hist(sino_corrected.shape[1] / 2 - vo_centers)
    plt.title("Гистограмма предсказанных сдвигов относительно центра")
    plt.xlabel('Сдвиг')
    plt.ylabel('Количество')
    plt.show()
    return vo_centers
    

def test_crops_flips(func, sino_corrected, angles, flip_prob = 0.5, crop_right = 0.005, crop_left = 0.005, n_experiments = 12, *args, **kwargs):
    '''
    '''
    vo_centers = []
    for i in tqdm(range(n_experiments)):
        shift_left = np.random.uniform(0, crop_left)
        shift_right = np.random.uniform(0, crop_right)
        sino_cropped = sino_corrected[:, int(shift_left * sino_corrected.shape[1]):int((1 - shift_right) * sino_corrected.shape[1])]
        result_transform = lambda x: x + int(shift_left * sino_corrected.shape[1])
        if np.random.rand() > 1 - flip_prob:
            sino_cropped = sino_cropped[:, ::-1]
            result_transform = lambda x: sino_cropped.shape[1] - x + int(shift_left * sino_corrected.shape[1])
        sino_slices = prepare_slices(sino_cropped, angles)
        vo_centers.append(result_transform(func(sino_slices, *args, **kwargs)))
    vo_centers = np.array(vo_centers)
    print('\nmean:', np.mean(sino_corrected.shape[1] / 2 - vo_centers), 'std:', np.std(sino_corrected.shape[1] / 2 - vo_centers))
    plt.hist(sino_corrected.shape[1] / 2 - vo_centers)
    plt.title("Гистограмма предсказанных сдвигов относительно центра")
    plt.xlabel('Сдвиг')
    plt.ylabel('Количество')
    plt.show()
    return vo_centers


def hard_test(func, sino_corrected, angles, flip_prob = 0.5, crop_right = 0.005, crop_left = 0.005, n_experiments = 12, n_turns=8, *args, **kwargs):
    '''
    '''
    vo_centers = []
    for i in tqdm(range(n_experiments)):
        shifts = np.random.uniform(-np.pi * 1 / 2, np.pi * 1 / 2, size=n_turns)
        shift_left = np.random.uniform(0, crop_left)
        shift_right = np.random.uniform(0, crop_right)
        sino_cropped = sino_corrected[:, int(shift_left * sino_corrected.shape[1]):int((1 - shift_right) * sino_corrected.shape[1])]
        result_transform = lambda x: x + int(shift_left * sino_corrected.shape[1])
        if np.random.rand() > 1 - flip_prob:
            sino_cropped = sino_cropped[:, ::-1]
            result_transform = lambda x: sino_cropped.shape[1] - x + int(shift_left * sino_corrected.shape[1])
        sino_slices = prepare_slices(sino_cropped, angles, shifts, n_turns=n_turns)
        vo_centers.append(result_transform(func(sino_slices, *args, **kwargs)))
    vo_centers = np.array(vo_centers)
    print('\nmean:', np.mean(sino_corrected.shape[1] / 2 - vo_centers), 'std:', np.std(sino_corrected.shape[1] / 2 - vo_centers))
    plt.hist(sino_corrected.shape[1] / 2 - vo_centers)
    plt.title("Гистограмма предсказанных сдвигов относительно центра")
    plt.xlabel('Сдвиг')
    plt.ylabel('Количество')
    plt.show()
    return vo_centers