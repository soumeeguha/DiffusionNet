import numpy as np
import matplotlib.image as io
import matplotlib.pyplot as plt
from scipy import signal
from skimage.restoration import denoise_nl_means, estimate_sigma
from numpy import linalg as LA
from scipy.ndimage.interpolation import rotate
# from sklearn.neighbors import KernelDensity
import itertools
import cv2
from operator import itemgetter


def get_centroid(image):   
    filter_size = 11
    filter = np.ones((filter_size,filter_size))
    filter_offset = (filter_size-1)/2
    filter_img = signal.convolve2d(image, filter, mode='same', boundary='fill', fillvalue=0)
    max_indices = np.where(filter_img == filter_img.max())
    max_indices = list(max_indices)
    list_indices = []
    for i in max_indices:
        list_indices.append(list(i))
    if len(list_indices[0])>1:
        max_indices = sorted(list_indices, key=itemgetter(0, 1))
    else:
        max_indices = list_indices
    
    if len(max_indices[0])==1:
        index_0 = max_indices[0][0]
        index_1 = max_indices[1][0]
    else:
        if len(max_indices)==2:
            index_0 = max_indices[0][0]
            index_1 = max_indices[0][1]
        else:
            index_0 = max_indices[len(max_indices)//2][0]
            index_1 = max_indices[len(max_indices)//2][1]

    return index_0, index_1


def rotate_matrix(mat, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return 

def get_rotated_img(x_centre, y_centre, theta, img, max_freq_intensity):
    
    deformed_img = np.ones((img.shape[0], img.shape[1]))*max_freq_intensity
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            xc = x - x_centre
            yc = y - y_centre
            x_new = int(np.round(xc*np.cos(theta) - yc*np.sin(theta) + img.shape[0]//2))
            y_new = int(np.round(xc*np.sin(theta) + yc*np.cos(theta) + img.shape[1]//2))
            if x_new>=0 and x_new<img.shape[0] and y_new>=0 and y_new<img.shape[1]:
                if img[x, y]>0:
                    deformed_img[x_new][y_new] = img[x, y]
                
                
    return deformed_img


def segment_img(im, diff):  
    hist = plt.hist(im.ravel(), bins = 50, range = (im.min(), im.max()))
    indices = hist[0][np.argmax(hist[0]):]
    frequency = hist[1][np.argmax(hist[0]):]
    first_zero = np.where(indices == 0)[0][0]
    if diff < 4:
        thresh1 = 0.01
    else: thresh1 = 0.03

    if (np.sum(indices[first_zero:])/np.sum(indices)) < thresh1:
        for i in range(1, len(indices)):
            if (np.sum(indices[-i:])/np.sum(indices)) >= 0.09:
                index = i
                break
        # print(i)
        seg_im = (im>frequency[len(frequency) - i])*im
    elif (np.sum(indices[first_zero:])/np.sum(indices)) > 0.2:
        for i in range(1, len(indices)):
            if (np.sum(indices[-i:])/np.sum(indices)) >= 0.09:
                index = i
                break
        # print(i)
        seg_im = (im>frequency[len(frequency) - i])*im
    else:
        seg_im = (im>frequency[np.where(indices == 0)[0][0]])*im
    
    return seg_im, hist[1][np.argmax(hist[0])]


def get_aligned_img(image, index_0, index_1, max_freq_intensity):
    segmented_im, max_freq_intensity = segment_img(image, diff=10)
    (r, c) = np.where(segmented_im!=0)
    X = np.hstack((np.expand_dims(c, axis=1), np.expand_dims(r, axis=1)))
    X_mean = np.mean(X, axis = 0)
    X_centred = X - np.tile(X_mean, (np.shape(X)[0], 1))
    X_cov = np.cov(np.transpose(X_centred))

    value, vector = LA.eig(X_cov)
    max_variation = vector[:, np.argmax(value)]
    angle = np.arctan(max_variation[1]/max_variation[0])
    new = get_rotated_img(index_0, index_1, angle, image, max_freq_intensity)
    return new


num_files = 10
num_imgs = 600

root = '/mnt/data/Soumee/gahlmann_imgs/'

for file in range(num_files):
    print(file)
    for num_im in range(num_imgs):
        image = plt.imread(root + f'renamed_imgs/d{file+1}img/img{num_im + 1}.png')
        index_0, index_1 = get_centroid(image)
        segmented_im, max_freq_intensity = segment_img(image, file+1)
        new = get_aligned_img(image, index_0, index_1, max_freq_intensity)
        cv2.imwrite(root + f'rotated_imgs/d{file+1}img/img{num_im + 1}.png', new*255)