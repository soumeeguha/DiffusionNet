import os
import matplotlib.pyplot as plt
import cv2

for i in range(10):
    print(i)
    diff = i + 1
    os.chdir(f'/mnt/data/Soumee/gahlmann_imgs/selected_imgs/d{diff}img/')
    # print(os.getcwd())
    # print(os.listdir())
    
    # for count, f in enumerate(os.listdir()):
    #     f_name, f_ext = os.path.splitext(f)
    #     f_name = "img" + str(count)
    
    #     new_name = f'{f_name}{f_ext}'
    #     os.rename(f, new_name)

    count = 1
    for im_name in os.listdir():
        im = plt.imread(f'/mnt/data/Soumee/gahlmann_imgs/selected_imgs/d{diff}img/' + im_name)
        # plt.imsave(f'/mnt/data/Soumee/gahlmann_imgs/renamed_imgs/d{diff}img/img{count}.png', im)
        cv2.imwrite(f'/mnt/data/Soumee/gahlmann_imgs/renamed_imgs/d{diff}img/img{count}.png', im*255)
        count += 1