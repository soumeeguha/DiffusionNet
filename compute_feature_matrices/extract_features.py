import numpy as np
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
from skimage.feature import hog
from skimage import data, exposure
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import cv2

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

def get_thresholded_img(hist, denoise):
    thresh = np.ceil(hist[1][np.where(hist[0] == hist[0].max())[0][0]])
    threshold_img = denoise.copy()
    threshold_img = np.squeeze((threshold_img > thresh)*denoise)

    return threshold_img

def get_roi(filter_size, window, denoise):
    filter = np.ones((filter_size,filter_size))
    filter_offset = (filter_size-1)/2
    filter_img = signal.convolve2d(denoise, filter, mode='same', boundary='fill', fillvalue=0)

    index = np.where(filter_img == filter_img.max())
    image_cropped = denoise[index[0][0] -window:index[0][0] + window, index[1][0]-window:index[1][0] + window]
    (row, col) = np.where(image_cropped > np.median(image_cropped) + 5)

    return row, col, image_cropped

def get_fft_magnitude(image_cropped):
    # print(np.shape(image_cropped))
    f = np.fft.fft2(image_cropped)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    magnitude_spectrum = magnitude_spectrum.ravel()
    return magnitude_spectrum.tolist()

def get_wavelet_features(image_cropped):
    coeffs = pywt.dwt2(image_cropped, 'haar')
    cA, (cH, cV, cD) = coeffs
    wavelets = np.array(cA.flatten().tolist() + cH.flatten().tolist() + cV.flatten().tolist() + cD.flatten().tolist())
    return wavelets.tolist()


def get_hog(im):
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
    return hog_image.flatten().tolist()


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    filter_vals = []
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        filter_vals.extend(filtered.flatten().tolist())
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return filter_vals



def get_image_features(filename, kernels):
    image = plt.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoise = get_denoised_img(image)
    try:
        hist = plt.hist(denoise.ravel(), bins = 50, range = (denoise.min(), denoise.max()))
        row, col, image_cropped = get_roi(11, 17, denoise)
        fft_magnitude_spectrum = get_fft_magnitude(image_cropped)
        wavelet_features = get_wavelet_features(image_cropped)
        features = fft_magnitude_spectrum + wavelet_features 
        normalized_features = features/np.abs(features).sum()
    except:
        # print(np.shape(image_cropped))
        # print(filename)
        normalized_features = None 
        image_cropped = None
    
    # hog_image = get_hog(image_cropped)
    # filter_comps = compute_feats(image_cropped, kernels)

    # print(f'FFT: {len(fft_magnitude_spectrum)}')
    # print(f'Wavelet: {len(wavelet_features)}')
    # print(f'HOG: {len(hog_image)}')
    # print(f'Texture: {len(filter_comps)}')

    
    return normalized_features, image_cropped