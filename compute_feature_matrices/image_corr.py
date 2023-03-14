import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import seaborn as sns

root = '/mnt/data/Soumee/gahlmann_imgs/renamed_imgs/'
num_img = 600
num_img_compared = 300
num_class = 10
img_indices = sorted(random.sample(range(num_img), num_img_compared))

corr_array = np.zeros((num_class, num_class))

def normalize_im(im):
    max_px = np.max(im)
    return im/max_px

def get_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean)*(y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2))*(np.sum((y - y_mean)**2))

    # print(numerator, denominator)
    return numerator/denominator

for i in range(num_class):
    print(f'i = {i}')
    for j in range(i, num_class):
        corr = 0
        num= 0
        for idx_i in img_indices:
            im_i = Image.open(root + f'd{i+1}img/img{idx_i + 1}.png').convert('L')
            im_i = normalize_im(im_i)
            for idx_j in img_indices:
                im_j = Image.open(root + f'd{j+1}img/img{idx_j + 1}.png').convert('L')
                im_j = normalize_im(im_j)
                corr += get_correlation(np.ravel(im_i), np.ravel(im_j))
                num+=1
        corr_array[i, j] = corr/(num_img_compared*num_img_compared)
        corr_array[j, i] = corr_array[i, j]

df = pd.DataFrame(
    corr_array,
    columns= [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
    ]
)

df = df.rename(index = {
    0: '1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10'
})
print(df)
print(num)

plt.figure(figsize =(10, 8))
heat = sns.heatmap(df, annot = True, cmap="Blues")
plt.tight_layout()
plt.savefig('heat.png', dpi = 600)
