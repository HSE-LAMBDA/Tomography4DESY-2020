from reconstruct import iradon_centered
from skimage.filters import sobel_h, sobel_v
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def image_entropy(reco, eps=1e-12):
    return np.sum(reco * np.log(reco + eps))


def inv_image_gradient(reco, eps=1e-12):
    sobelx = sobel_v(reco)
    sobely = sobel_h(reco)
    grad_norm = (sobelx ** 2 + sobely ** 2) ** (1 / 2)
    return grad_norm.sum()


def run_method_on_set(sino, angles, shift=15, use_spline_minima=True, method=inv_image_gradient, reconstructor=iradon_centered, verbose=True):
    best_id = 0
    temp_loss = []
    for i in tqdm(range(-shift, shift), ):
        reco = reconstructor(sino, angles, center=sino.shape[1] / 2. + i)
        temp_sum = method(reco)
        temp_loss.append(temp_sum)
    if use_spline_minima:
        x_axis = np.arange(-shift, shift)
        f = InterpolatedUnivariateSpline(x_axis, np.array(temp_loss), k=4)
        cr_pts = f.derivative().roots()
        cr_pts = np.append(cr_pts, (x_axis[0], x_axis[-1]))  # also check the endpoints of the interval
        cr_vals = f(cr_pts)
        min_index = np.argmin(cr_vals)
        max_index = np.argmax(cr_vals)
        min_point = cr_pts[min_index]
    else:
        min_point = np.argmin(temp_loss) - shift
    if verbose:
        print('predict:', min_point)
        plt.plot(np.arange(-shift, shift), temp_loss)
        plt.show()
    return min_point


def run_method_diff(sino, angles, start_point=0., eps=0.25, iters=20, step=0.5, step_size='gradient', method=inv_image_gradient, reconstructor=iradon_centered, verbose=True):
    shift = start_point
    temp_loss = []
    for i in range(iters):
        print(sino.shape[1] / 2. + shift - eps, sino.shape[1] / 2. + shift + eps)
        reco_left = reconstructor(sino.copy(), angles, center=sino.shape[1] / 2. + shift - eps)
        reco_right = reconstructor(sino.copy(), angles, center=sino.shape[1] / 2. + shift + eps)
        sum_left = method(reco_left)
        sum_right = method(reco_right)
        gradient = (sum_right - sum_left) / (2 * eps)
        update = gradient * step
        if step_size == 'fixed':
            update = step * np.sign(gradient)
        if verbose:
            print('iter:', i, 'sums:', sum_left, sum_right, 'gradient:', gradient, 'new_shift:', shift, 'update:', update)
        if update < eps:
            break
        shift += update
    return shift