from PIL import Image
import matplotlib.pyplot as plt
import glob
import cv2

root = '/mnt/data/Soumee/unconfined_diffusion/'
num_files = 10

def get_image_list(images):
    image_list = []
    files = sorted(glob.glob(images))
    image_list.extend(files)
    return image_list

for i in range(num_files):
    print(i)
    image_features = []
    images = root + f'selected_imgs/d{i+1}img/*.tif'
    for filename in get_image_list(images):
        image = plt.imread(filename)
        # print(filename)
        img_name = filename.split('/')[-1].split('.')[0]
        cv2.imwrite(root + f'png_imgs/d{i+1}img/' + img_name + '.png', image)