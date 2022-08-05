import os
import shutil
from PIL import Image
import numpy as np
import scipy.ndimage
import os
import scipy.io
import h5py
import pdb

def save_npy(source_dir, target_dir):
    if not os.path.isdir(source_dir):
        os.makedirs(source_dir)
    nyud_file_path = os.path.join(source_dir, 'nyu_depth_v2_labeled.mat')
    splits_file_path = os.path.join(source_dir, 'splits.mat')

    print("Loading dataset: NYU Depth V2")
    nyud_dict = h5py.File(nyud_file_path, 'r')
    splits_dict = scipy.io.loadmat(splits_file_path)

    images = np.asarray(nyud_dict['images'], dtype=np.float32)
    depths = np.asarray(nyud_dict['depths'], dtype=np.float32)

    images = images.swapaxes(2, 3)
    depths = np.expand_dims(depths.swapaxes(1, 2), 1)

    indices = splits_dict['testNdxs'][:, 0] - 1

    images = np.take(images, indices, axis=0)
    depths = np.take(depths, indices, axis=0)

    npy_folder = os.path.join(target_dir, 'npy')
    os.makedirs(npy_folder)

    np.save(os.path.join(npy_folder, 'images.npy'), images)
    np.save(os.path.join(npy_folder, 'depths.npy'), depths)

if __name__ == '__main__':
    save_npy('C:/Users/psiml8/Documents/GitHub/PSIML',
             'C:/Users/psiml8/Documents/GitHub/PSIML')