from cgi import test
import numpy as np
import matplotlib.image as io
import matplotlib.pyplot as plt
from scipy import signal
from skimage.restoration import denoise_nl_means, estimate_sigma
from numpy import linalg as LA
from scipy.ndimage.interpolation import rotate
from sklearn.neighbors import KernelDensity
import itertools
from extract_features import *
import pywt
import glob
import pandas as pd
import torch
from PIL import Image
import cv2
from torchvision import transforms


root = '/mnt/data/Soumee/unconfined_diffusion/'
num_files = 10

def get_image_list(images):
    image_list = []
    files = sorted(glob.glob(images))
    image_list.extend(files)
    return image_list

def prepare_kernels():
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                            sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels

kernels = prepare_kernels()

all_features = []
for i in range(num_files):
    print(i)
    image_features = []
    images = root + f'selected_imgs/d{i+1}img/*.tif'
    img_num = 1
    for filename in get_image_list(images):
        features, image_cropped = get_image_features(filename, kernels)
        if (features is not None) and (image_cropped is not None):
            # img_name = filename.split('/')[-1].split('.')[0]
            # cv2.imwrite(root + f'cropped_imgs/d{i+1}img/' + img_name + '.png', image_cropped)
            # cv2.imwrite(root + f'cropped_imgs/d{i+1}img/img{img_num}.png', image_cropped)
            image_features.append(features)
            img_num += 1
    df = pd.DataFrame(image_features)
    # df.to_csv(root + f'features/features{i+1}.csv', index= False)

