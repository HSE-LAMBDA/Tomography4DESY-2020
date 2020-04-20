import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
import radontea
from scipy.interpolate import interp1d
import math
from tqdm import tqdm
import tensorflow as tf



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



def np_iradon_custom(sino, angles, filtering=True, circle=True):
    ln = sino.shape[1]
    # filetring:
    if filtering:
        kx = 2 * np.pi * np.abs(np.fft.fftfreq(ln))
        kx = kx[None, :]
        projection = np.fft.fft(sino, axis=0) * kx
        sino_filtered = np.real(np.fft.ifft(projection, axis=0))
        # Resize filtered sinogram back to original size
        sino = sino_filtered[:, :ln]
    
    # main process:
    fourier_sino = np.fft.rfft(sino, axis=0)
    wfilt = np.linspace(0, fourier_sino.shape[0], fourier_sino.shape[0])[:,np.newaxis]
    #wfilt *= (np.cos(wfilt * np.pi / 2))**2
    backprojections = np.fft.irfft(fourier_sino * wfilt, axis=0)
    result = np.zeros(dtype=float, shape=backprojections.shape[:1] * 2)
    size = result.shape[0]
    xx, yy = np.meshgrid(*[np.linspace(-(size - 1) / 2., (size - 1) / 2., size)] * 2)
    for th, backproj in zip(angles, backprojections.T):
        th *= np.pi / 180.
        coords = xx * np.cos(th) + yy * np.sin(th) + (size - 1) / 2
        ccoords = np.ceil(coords).astype(int)
        fcoords = np.floor(coords).astype(int)
# #         print(th, coords, ccoords, fcoords)
        cmask = (ccoords >= 0) & (ccoords < size)
        fmask = (fcoords >= 0) & (fcoords < size)
        wc = 1 - (ccoords - coords)
        wf = 1 - (coords - fcoords)
#         print(wc, wf)
        result += backproj[np.where(cmask, ccoords, 0)] * wc + \
                    backproj[np.where(fmask, fcoords, 0)] * wf
    if circle:
        out_reconstruction_circle = (xx ** 2 + yy ** 2) >= ((size - 1) / 2) ** 2
        result[out_reconstruction_circle] = 0.
    return result / len(angles) / (2. * np.pi)



def tf_iradon_custom(sino, angles, filtering=True, circle=True):
    sino = tf.convert_to_tensor(sino, dtype=tf.float64)
    angles = tf.convert_to_tensor(angles * np.pi / 180., dtype=tf.float64)
    
    if filtering:
        ln = sino.shape[1]
        kx = 2 * np.pi * np.abs(np.fft.fftfreq(ln))
        kx = kx[None, :]
        kx = tf.convert_to_tensor(kx, dtype=tf.float64)
        sino_prepared = tf.transpose(tf.cast(sino, tf.complex128))
        projection = tf.transpose(tf.signal.fft(sino_prepared)) * tf.cast(kx, tf.complex128)
        sino_filtered = tf.transpose(tf.math.real(tf.signal.ifft(tf.transpose(projection))))
        # Resize filtered sinogram back to original size
        sino = sino_filtered[:, :ln]
    
    # main process:
    fourier_sino = tf.transpose(tf.signal.rfft(tf.transpose(sino)))  # maybe 2d?
    shape = tf.shape(fourier_sino)[0]
    wfilt = tf.linspace(0., tf.cast(shape, dtype=tf.float64), shape)[:, None]
    wfilt = tf.cast(wfilt, tf.complex128)
    backprojections = tf.transpose(tf.signal.irfft(tf.transpose(fourier_sino * wfilt)))
    size = tf.shape(backprojections)[0]
    result = tf.zeros(dtype=tf.float64, shape=[size] * 2)
    xx, yy = tf.meshgrid(*[tf.linspace(-(size - 1) / 2, (size - 1) / 2, size)] * 2)
    backprojections = tf.transpose(backprojections)
    for th, backproj in zip(angles, backprojections):
        coords = tf.cast(xx, tf.float64) * tf.cos(th) + tf.cast(yy, tf.float64) * tf.sin(th) + (tf.cast(size, tf.float64) - 1.) / 2.
        ccoords = tf.cast(tf.math.ceil(coords), tf.int32)
        fcoords = tf.cast(tf.math.floor(coords), tf.int32)
        cmask = tf.math.logical_and(ccoords >= 0, ccoords < size)
        fmask = tf.math.logical_and(fcoords >= 0, fcoords < size)
        wc = 1. - (tf.cast(ccoords, tf.float64) - coords)
        wf = 1. - (coords - tf.cast(fcoords, tf.float64))
        result += tf.gather(backproj, tf.where(cmask, ccoords, 0)) * wc + \
                tf.gather(backproj, tf.where(fmask, fcoords, 0)) * wf
    result = result.numpy() / angles.shape[0] / (2. * np.pi)
    if circle:
        out_reconstruction_circle = (xx.numpy() ** 2 + yy.numpy() ** 2) >= ((size.numpy() - 1) / 2) ** 2
        result[out_reconstruction_circle] = 0.
    return result



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
    elif lib == 'custom_np':
        reco = np_iradon_custom(fixed_image.T, angles * 180 / np.pi)
    elif lib == 'custom_tf':
        reco = tf_iradon_custom(fixed_image.T, angles * 180 / np.pi)
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