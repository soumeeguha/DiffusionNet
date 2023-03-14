from selectors import EpollSelector
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from loss_definitions import *
from get_dataset import *
from models import *
from train_val_test import train, val, test, test_real_imgs
from train_autoencoder import train_ae
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")


# num_files = 20
num_files = [1, 6]



num_images = 300
train_val_ratio = 0.8
num_worker_train = 2
batch_size = 8
num_epochs = 5
ngpu = 2

root = '/mnt/data/Soumee/gahlmann_imgs/'
train_indices = sorted(random.sample(range(num_images), int(train_val_ratio*num_images)))
val_indices = sorted(list(set(range(num_images)) - set(train_indices)))
image_size = 30

transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(image_size),
                                transforms.RandomHorizontalFlip(p = 0.4),
                                transforms.RandomVerticalFlip(p = 0.4),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5),
                           ])

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    
    if type(m) == nn.Linear:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

sim_train_img, val_img = get_train_test_image_lists(root, train_indices, val_indices, num_files)
sim_train_df, sim_train_labels, val_df, val_labels = get_train_test_df_labels(root, train_indices, val_indices, num_files)

real_imgs_root = '/mnt/data/Soumee/gahlmann_imgs/augment_domain_adaptation/augmented_train_test/1/'
test_folders = ['test/']
train_folders = ['train0.5/', 'train5.5/']
real_train_img, test_img = get_real_augmented_img_list(real_imgs_root, train_folders), get_real_augmented_img_list(real_imgs_root, test_folders)
real_train_df, real_train_labels, test_df, test_labels = get_real_augmented_train_test_df_labels(real_imgs_root, num_files)

# print(test_img)

test_labels = []
test_labels.extend(38*[num_files[0]])
test_labels.extend(6*[num_files[1]])

train_img, train_labels = [], []
train_img.extend(sim_train_img)
train_img.extend(real_train_img)

train_labels.extend(sim_train_labels)
train_labels.extend(real_train_labels)


# for idx in range(len(train_labels)):
#     perturbation = random.uniform(0, 1)*0.1
#     if train_labels[idx] == 0:
#         train_labels[idx] += perturbation
#     else:
#         train_labels[idx] -= perturbation

train_df = pd.concat([sim_train_df, real_train_df])

# print(test_img)

ds_train = get_Dataset(train_df, train_labels, train_img, transform)
ds_val = get_Dataset(val_df, val_labels, val_img, transform)
ds_test = get_Dataset(test_df, test_labels, test_img, transform)

# print(len(ds_train[0][0]))
dl_train, dl_val = setup_dataloader(batch_size, num_worker_train, ds_train, ds_val)
dl_test, _ = setup_dataloader(len(test_labels), num_worker_train, ds_test, None)

# device = torch.device("cuda:1")
device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

learning_rate = 0.00005
conv_ch_features = 40
conv_ch_img = 20
map_size = 150
print(ds_train[0][0].size()[-1])
print(ds_train[0][1].size())

features_in = ds_train[0][0].size()[-1] 
model = NetworkAttentionSpChImproved(device, ds_train[0][0].size()[-1], image_size, len(num_files), conv_ch_features, conv_ch_img, map_size).to(device)
model.apply(weights_init_uniform_rule)

optimizerC = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999))

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#
#------------------
# criterionC = torch.nn.CrossEntropyLoss()
#------------------
criterionC = MaxEntropyAttnMaps()

schedulerC = torch.optim.lr_scheduler.StepLR(optimizerC, step_size=10, gamma=0.5)

for epoch in range(num_epochs):
    train(epoch, dl_train, model, device, criterionC, optimizerC, schedulerC)
    val(epoch, dl_val, model, device, criterionC)

targets, preds, acc, y_pred_tags = test(epoch, dl_val, model, device, criterionC)
df = pd.DataFrame({'targets': targets, 'pred': y_pred_tags})
df['score'] = np.where(df.targets == df.pred, 1, 0)


print(df.groupby(['targets']).mean()['score'], df.groupby(['targets']).mean()['score'].mean())
# print(df)
# print(targets, y_pred_tags)

y_pred_tags, test_targets, predicted = test_real_imgs(dl_test, model, device, criterionC)

pred_lbl = y_pred_tags.tolist()
target_lbl = test_targets.tolist()


for i in range(len(pred_lbl)):
    if pred_lbl[i] == 0:
        pred_lbl[i] = num_files[0]
    else:
        pred_lbl[i] = num_files[1]

sum_lbl = 0
for i in range(len(pred_lbl)):
    print(pred_lbl[i], target_lbl[i])
    if pred_lbl[i] == target_lbl[i]:
        sum_lbl += 1
        
    
print(f'The accuracy on real images is: {100*sum_lbl/len(pred_lbl)}%')

# print(test_targets, predicted)