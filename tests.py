import numpy as np


def prepare_slices(sino, angles, shift_left=0, shift_right=0, n_turns=8):
    '''
    Prepare slices of sinogram for testing Vo's method
     :param sino: prepared sinogram
     :param angles: angles of taken sinogram
     :param shift_left: shift of angles for the left end of rotation
     :param shift_right: shift of angles for the left end of rotation
     :param n_turns: number of rotations of sinogram
     
     :return: prepared sino slices
    '''
    sino_slices = []
    for i in range(n_turns):
        if shift_right > 0 and i == n_turns - 1: continue  # skip not full sized 
        if shift_left < 0 and i == 0: continue             # skip not full sized 
        mask = ((i * np.pi + shift_left < angles.squeeze()) & (angles.squeeze() < (i + 1) * np.pi + shift_right))
        sino_slices.append(sino_corrected[mask])
    min_len = min(len(slice_) for slice_ in sino_slices)
    sino_slices = np.array([sino_slice[:min_len] for sino_slice in sino_slices]).transpose(1, 0, 2)
    return sino_slices