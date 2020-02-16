import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
import radontea
from scipy.interpolate import interp1d
import math


def get_from_mat(filename):
    ''' loads sino '''
    file = h5py.File(filename, 'r')
    key, = list(file.keys())
    return file.get(key)[:]


def correct_image(img, shifts, linear_interpolation=True, **args):
    ''' 
    Correct shifted image using linear interpolation, example:
     :param img: raw image
     :param shifts: np.array of desired shifts
     :param linear_interpolation: use linear interpolation or simple round
     **args for scipy.interpolate.interp1d, refer scipy.interpolate
    
    Example:
        sino_corrected_linear = correct_image(sino_raw, 1.5 * shifts.squeeze(), linear_interpolation=True, kind='cubic')
      
    :return: np.array corrected image
    '''
    shifts = shifts.squeeze()

    assert img.ndim == 2
    assert img.shape[0] == shifts.shape[0]

    margin_right = -int(np.floor(shifts.min()))
    margin_left  =  int(np.ceil (shifts.max()))
    assert margin_left > 0 and margin_right > 0

    result = np.zeros(shape=(img.shape[0], img.shape[1] + margin_left + margin_right), dtype=img.dtype)

    for i_row, (row, shift) in enumerate(zip(img, shifts)):
        if linear_interpolation:
            x = np.arange(len(row))
            f = interp1d(x - shift, row, **args)
            row_interpolated = f(np.clip(np.arange(-margin_left, img.shape[1] + margin_right), (x - shift).min(), (x - shift).max()))
            result[i_row] = row_interpolated
        else:
            shift = -int(np.round(shift))
            result[i_row][margin_left + shift : margin_left + shift + len(row)] = row

    return result[:,margin_left + margin_right : -(margin_left + margin_right)]



def iradon_centered(image, angles, center, kind='linear', lib='scipy', show=False):
    '''
    Apply inverse randon transform to image.
      :param image: corrected image
      :param angles: array of angles sino
      :param center: desired center of sinogram
      :param kind: kind of interpolation for scipy.interpolate.interp1d used in centering
      :param lib: scipy or radontea - desired lib for reconsctruction
      :param show: whether to show shifted sinogram and final result
    
    :return: reconstruction
    '''
    shift = image.shape[1] / 2 - center
    fixed_image = np.zeros((image.shape[0], 2 * math.ceil(np.abs(shift)) + image.shape[1]))
    for i_row, row in enumerate(image):
        x = np.arange(len(row))
        if shift < 0:
            f = interp1d(x - shift, row, kind=kind)
            row_interpolated = f(np.clip(np.arange(-shift, image.shape[1]), (x - shift).min(), (x - shift).max()))
            fixed_image[i_row, :len(row_interpolated)] = row_interpolated
        else:
            f = interp1d(x + shift, row, kind=kind)
            row_interpolated = f(np.clip(np.arange(0, image.shape[1] + shift), (x + shift).min(), (x + shift).max()))
            fixed_image[i_row, -len(row_interpolated):] = row_interpolated
    if show:
        plt.figure(figsize=(10, 12))
        plt.imshow(fixed_image)
        plt.title('prepared sino with shift: ' + str(shift))
        plt.show()
    if lib == 'radontea':
        reco = radontea.backproject(fixed_image, angles)
    elif lib == 'scipy':
        reco = iradon(fixed_image.T, angles * 180 / np.pi)
    else:
        raise NotImplemented('This lib is unknown')
    if show:
        plt.figure(figsize=(15, 15))
        plt.imshow(reco, vmax=0.001)
        plt.show()
    return reco


def find_visual_best(centers, sino, angles, from_x=0, from_y=0, to_x=None, to_y=None, ncols=2, **kwargs):
    '''
    Plotting part of reconstrunctions with specified center for visiual check of quality.
    
        :param centers: iterable of desired centers
        :param sino: prepared for iradon_centered sino
        :param angles: angles for iradon_centered
        :param from_x: desired crop start for axis 0, zero by default
        :param from_y: desired crop start for axis 1, zero by default
        :param to_y: desired crop end for axis 0, None by default, which means reco.shape[0]
        :param to_y: desired crop end for axis 1, None by default, which means reco.shape[1]
        :param ncols: number of columns to be plotted

    :return: plt.figure
    '''
    
    nrows = math.ceil(len(centers) / ncols)
    f, axs = plt.subplots(nrows, ncols, figsize=(15, 25))
    for i, center in enumerate(tqdm(centers)):
        kwargs['show'] = kwargs.get('show', False)
        reco = iradon_centered(sino, angles, center=sino.shape[1] / 2 + center, **kwargs)
        row = i // ncols
        col = i % ncols
        if to_x is None:
            to_x = reco.shape[0]
        if to_y is None:
            to_y = reco.shape[1]
        axs[row][col].imshow(reco[from_x:to_x,from_y:to_y])
        axs[row][col].set_title('center: ' + str(center))
    return f