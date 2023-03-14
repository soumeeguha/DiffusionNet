import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

import pandas as pd

root = '/mnt/data/Soumee/gahlmann_imgs/renamed_imgs/'
files = 10
tot_images = 600

radon_features = []

for file in range(files):
    radon_features = []
    print(file)
    for im_num in range(tot_images):

        im = plt.imread(root + f'd{file+1}img/img{im_num+1}.png')
        hist = plt.hist(im.ravel(), bins = 50, range = (im.min(), im.max()))
        indices = hist[0][np.argmax(hist[0]):]
        frequency = hist[1][np.argmax(hist[0]):]
        diff = 1
        first_zero = np.where(indices == 0)[0][0]
        if diff < 4:
            thresh1 = 0.01
        else: thresh1 = 0.03
        if np.sum(indices[first_zero:])/np.sum(indices) < thresh1:
            for i in range(1, len(indices)):
                if (np.sum(indices[-i:])/np.sum(indices)) >= 0.09:
                    index = i
                    break

            intensity = np.sum((im>frequency[len(frequency) - index])*im)
            seg_im = (im>frequency[len(frequency) - index])*im
        elif np.sum(indices[first_zero:])/np.sum(indices) > 0.2:
            for i in range(1, len(indices)):
                if (np.sum(indices[-i:])/np.sum(indices)) >= 0.09:
                    index = i
                    break

            intensity = np.sum((im>frequency[len(frequency) - index])*im)
            seg_im = (im>frequency[len(frequency) - index])*im
        else:
            intensity = np.sum((im>frequency[np.where(indices == 0)[0][0]])*im)
            seg_im = (im>frequency[np.where(indices == 0)[0][0]])*im

        theta = np.linspace(0., 180., max(seg_im.shape), endpoint=False)
        sinogram = radon(seg_im, theta=theta)

        radon_features.append(np.array(sinogram.tolist()).ravel().tolist())

    df = pd.DataFrame(radon_features)
    df.to_csv(root + f'radon_features/features{file+1}.csv', index= False)
