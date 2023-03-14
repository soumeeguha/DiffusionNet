import torch
import torch.nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.functional import normalize
from PIL import Image
import glob
import matplotlib.pyplot as plt 

def get_real_augmented_img_list(root, folders):
    image_list = []
    for folder in folders:
        image_list.extend(sorted(glob.glob(root + folder + '*.png')))
    # print(len(image_list))

    return image_list


def get_real_augmented_train_test_df_labels(root, numFiles):
    train = pd.read_csv(root + 'train_features.csv')
    test = pd.read_csv(root + 'test_features.csv')

    print(f'Train DF Shape: {train.shape}, Test DF Shape: {test.shape}')

    train_labels, test_labels = [], []
    num = 0
    num_train, num_test = int((train.shape[0])/2), int((test.shape[0])/2)

    # print(num_train, num_test)
    for i in numFiles:
        
        train_labels.extend([num]*num_train)
        test_labels.extend([num]*num_test)
        num += 1

    return train, train_labels, test, test_labels

def get_train_test_image_lists(root, train_indices, test_indices, numFiles):
    train_img = []
    test_img = []
    # str = '/home/tiffany/projects/gahlmann_lab/codes/image_classification/test_imgs/'
    for i in numFiles:
        # im = Image.open(root + f'rotated_imgs/d{i}img/img1.png')
        # plt.imsave(str + f'im{i}.png', im)

        for index in train_indices:
            train_img.append(root + f'renamed_imgs/d{i}img/img{index+1}.png')
        for index in test_indices:
            test_img.append(root + f'renamed_imgs/d{i}img/img{index+1}.png')

    return train_img, test_img

def get_train_test_df_labels(root, train_indices, test_indices, numFiles):
    train = pd.DataFrame()
    test = pd.DataFrame()
    train_labels, test_labels = [], []
    num = 0
    for i in numFiles:
        df = pd.read_csv(root + f'renamed_imgs/features/features{i}.csv')
        print('df shape: ', df.shape)
        train_labels.extend([num]*df[df.index.isin(train_indices)].shape[0])
        test_labels.extend([num]*df[df.index.isin(test_indices)].shape[0])
        train = pd.concat([train, df[df.index.isin(train_indices)]])
        test = pd.concat([test, df[df.index.isin(test_indices)]])
        num += 1

    return train, train_labels, test, test_labels


class get_Dataset(Dataset):
    def __init__(self, df, labels, image_list, transform = None):
        super().__init__()
        self.df = torch.tensor(df.values)
        self.image_list = image_list
        self.labels = torch.tensor(labels)
        self.transform = transform
        self.toTensor = transforms.ToTensor()


    def __len__(self):
        return self.df.size()[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        features = self.df[index, :][:2312]
        label = self.labels[index]
        # print('index: ', index)
        # print(self.image_list[index])
        image = self.toTensor(Image.open(self.image_list[index])).type(torch.float)/255


        if self.transform is not None:
            features = normalize(features, p=2.0, dim = 0)
            image = self.transform(image)

        dataset = (
            features,
            image, 
            label
        )

        return dataset


class get_Dataset_Finetuned_models(Dataset):
    def __init__(self, df, labels, image_list, transform = None):
        super().__init__()
        self.df = torch.tensor(df.values)
        self.image_list = image_list
        self.labels = torch.tensor(labels)
        self.transform = transform
        self.toTensor = transforms.ToTensor()


    def __len__(self):
        return self.df.size()[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        features = self.df[index, :][:2312]
        label = self.labels[index]
        image = self.toTensor(Image.open(self.image_list[index])).type(torch.float)/255
        image = torch.cat((image, image, image), 0)

        if self.transform is not None:
            features = normalize(features, p=2.0, dim = 0)
            image = self.transform(image)

        dataset = (
            image, 
            label
        )

        return dataset


def setup_dataloader(batch_size, num_worker_train, train_ds, test_ds=None):
    """Set's up dataloader"""

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=num_worker_train,
        pin_memory=True)

    if test_ds is not None:

        test_dl = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=num_worker_train,
            pin_memory=False)
    else:

        test_dl = None

    return train_dl, test_dl