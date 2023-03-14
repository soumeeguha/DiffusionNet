from operator import index
import pandas as pd


root = '/mnt/data/Soumee/gahlmann_imgs/renamed_imgs/'
num_files = 10

for i in range(num_files):
    print(i)
    df1 = pd.read_csv(root + f'features/features{i+1}.csv')
    df2 = pd.read_csv(root + f'radon_features/features{i+1}.csv')
    pd.concat([df1, df2], axis = 1).to_csv(root + f'ft_wt_radon_features/features{i+1}.csv', index = False)


