import numpy as np
import glob
import matplotlib.image as io
import matplotlib.pyplot as plt
from scipy import signal
from skimage.restoration import denoise_nl_means, estimate_sigma
from numpy import linalg as LA
from scipy.ndimage.interpolation import rotate
from sklearn.neighbors import KernelDensity
import itertools
from scipy import ndimage as ndi
import pywt
import cv2
from skimage.feature import hog
from skimage import data, exposure
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def denoise_non_local_means(image):
    sigma_est = np.mean(estimate_sigma(image))
    # print(f'estimated noise standard deviation = {sigma_est}')

    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6)

    denoise = denoise_nl_means(image, h=2 * sigma_est, preserve_range=True, fast_mode=False,
                            **patch_kw)

    return sigma_est, denoise

def get_denoised_img(image):
    denoise = image.copy()
    noise_prev = 10
    std_noise = 1
    while (std_noise>=0.01) and (np.abs(noise_prev-std_noise)>0.01):
        noise_prev = std_noise
        std_noise, denoise = denoise_non_local_means(denoise)
        

    return denoise

def get_roi(filter_size, window, denoise):
    filter = np.ones((filter_size,filter_size))
    filter_offset = (filter_size-1)/2
    filter_img = signal.convolve2d(denoise, filter, mode='same', boundary='fill', fillvalue=0)

    index = np.where(filter_img == filter_img.max())
    image_cropped = denoise[index[0][0] -window:index[0][0] + window, index[1][0]-window:index[1][0] + window]
    (row, col) = np.where(image_cropped > np.median(image_cropped) + 5)

    return row, col, image_cropped


def get_cropped_denoised(filename):
    image = plt.imread(filename)
    denoise = np.nan_to_num(get_denoised_img(image))

    hist = plt.hist(denoise.ravel(), bins = 50, range = (denoise.min(), denoise.max()))
    row, col, image_cropped = get_roi(11, 17, denoise)
    
    
    return image_cropped

root = '/mnt/data/Soumee/gahlmann_imgs/experimental_imgs/test_imgs/*.png'

# num_file = 17

def get_image_list(images):
    image_list = []
    files = sorted(glob.glob(images))
    image_list.extend(files)
    return image_list


num_im_written = 0
num_im_read = 1

print(get_image_list(root))
# for i in num_files:
#     print(i)
#     images = root + f'd{i}img/*.tif'
#     # for filename in get_image_list(images):
#     for j in range(num_im):
#         filename = root + f'd{i}img/img{j+1}.tif'
#         image_cropped = get_cropped_denoised(filename)
#         img_name = filename.split('/')[-1].split('.')[0]
#         cv2.imwrite(root + f'd{i}img/cropped/' + img_name + '.png', image_cropped)
        
# print(num_file)
# while num_im_written <= num_im:
#     filename = root + f'd{num_file}img/img{num_im_read}.tif'
#     num_im_read += 1

#     if num_im_read%1000 == 0:
#         print(f'Images read: {num_im_read}, Images processed: {num_im_written}')

#     try:
#         image_cropped = get_cropped_denoised(filename)
#         cv2.imwrite(root + f'd{num_file}img/cropped/img{num_im_written}.png', image_cropped)
#         num_im_written += 1
#     except:
#         pass

