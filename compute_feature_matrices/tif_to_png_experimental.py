from PIL import Image
import matplotlib.pyplot as plt
import glob
import cv2

def get_image_list(images):
    image_list = []
    files = sorted(glob.glob(images))
    # print(files)
    image_list.extend(files)
    return image_list

root = '/mnt/data/Soumee/gahlmann_imgs/'

folder = '30ms_dark_count/'
images = root + 'experimental_imgs/' + folder + '*.tif'
# print(get_image_list(images))
for filename in get_image_list(images):
    image = plt.imread(filename)
    img_name = filename.split('/')[-1].split('.')[0]
    cv2.imwrite(root + 'experimental_imgs_png/' + folder + img_name + '.png', image*255)
