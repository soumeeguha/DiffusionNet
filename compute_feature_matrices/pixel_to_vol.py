import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

root = '/mnt/data/Soumee/gahlmann_imgs/renamed_imgs/'

num_im = 600
num_file = 10

volume_array = np.zeros((num_im, num_file))
for file in range(num_file):
    print(file)
    volumes = []
    for im_num in range(num_im):
        im = plt.imread(root + f'd{file+1}img/img{im_num+1}.png')
        if file <= 5:
            hist = plt.hist(im.ravel(), bins = 50, range = (im.min(), im.max()))
        else:
            hist = plt.hist(im.ravel(), bins = 50, range = (im.min(), im.max()))


        indices = hist[0][np.argmax(hist[0]):]
        frequency = hist[1][np.argmax(hist[0]):]

        first_zero = np.where(indices == 0)[0][0]

        # if file > 5: 
        #     for i in range(1, len(indices)):
        #         if (np.sum(indices[-i:])/np.sum(indices)) >= 0.09:
        #             index = i
        #             break
        #     # frequency[np.where(indices == 0)[0][0]]

        #     intensity = np.sum((im>frequency[len(frequency) - index])*im)
        # else:
        #     if file < 5:
        #         thresh1 = 0.005
        #     else: thresh1 = 0.03
        #     if np.sum(indices[first_zero:])/np.sum(indices) < thresh1:
        #         for i in range(1, len(indices)):
        #             if (np.sum(indices[-i:])/np.sum(indices)) >= 0.09:
        #                 index = i
        #                 break

        #         intensity = np.sum((im>frequency[len(frequency) - index])*im)
        #         seg_im = (im>frequency[len(frequency) - index])*im
        #     elif np.sum(indices[first_zero:])/np.sum(indices) > 0.2:
        #         for i in range(1, len(indices)):
        #             if (np.sum(indices[-i:])/np.sum(indices)) >= 0.09:
        #                 index = i
        #                 break

        #         intensity = np.sum((im>frequency[len(frequency) - index])*im)
        #         seg_im = (im>frequency[len(frequency) - index])*im
        #     else:
        #         intensity = np.sum((im>frequency[np.where(indices == 0)[0][0]])*im)
        #         seg_im = (im>frequency[np.where(indices == 0)[0][0]])*im

        if file < 4:
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


        area = len(np.where(seg_im > 0)[0])/(np.shape(seg_im)[0]*np.shape(seg_im)[1])

        volume_array[im_num, file] = intensity*area
        # volumes.append(vol)

        if im_num%100 == 0:
            print(f'\tImages done: {im_num}')
    # volume_df[f'{num_file + 1}'] = volumes
print(np.shape(volume_array))
volume_df = pd.DataFrame(data = volume_array, columns= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
volume_df.to_csv('volume_df.csv', index=False)
print(volume_df.head())
print(volume_df.describe())