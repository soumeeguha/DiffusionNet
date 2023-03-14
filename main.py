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
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


# num_files = 20
num_files = [6, 10]



num_images = 600
train_val_ratio = 0.8
num_worker_train = 2
batch_size = 8
num_epochs = 15
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

train_img, val_img = get_train_test_image_lists(root, train_indices, val_indices, num_files)
train_df, train_labels, val_df, val_labels = get_train_test_df_labels(root, train_indices, val_indices, num_files)
ds_train = get_Dataset(train_df, train_labels, train_img, transform)
ds_val = get_Dataset(val_df, val_labels, val_img, transform)

# print(len(ds_train[0][0]))
dl_train, dl_val = setup_dataloader(batch_size, num_worker_train, ds_train, ds_val)

# device = torch.device("cuda:1")
device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

learning_rate = 0.00005
conv_ch_features = 40
conv_ch_img = 20
map_size = 150

# print(ds_train[0][0].size()[-1])
# print(ds_train[0][1].size())


features_in = ds_train[0][0].size()[-1] 
model = NetworkAttentionSpChImproved(device, ds_train[0][0].size()[-1], image_size, len(num_files), conv_ch_features, conv_ch_img, map_size).to(device)
model.apply(weights_init_uniform_rule)

optimizerC = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

criterionC = MaxEntropyAttnMaps()

schedulerC = torch.optim.lr_scheduler.StepLR(optimizerC, step_size=10, gamma=0.5)

for epoch in range(num_epochs):
    train(epoch, dl_train, model, device, criterionC, optimizerC, schedulerC)
    val(epoch, dl_val, model, device, criterionC)

targets, preds, acc, y_pred_tags = test(epoch, dl_val, model, device, criterionC)
print('F1 score: ', f1_score(targets.cpu().numpy(), y_pred_tags.cpu().numpy()))

# print(targets, preds)
df = pd.DataFrame({'targets': targets, 'pred': y_pred_tags})
df['score'] = np.where(df.targets == df.pred, 1, 0)


print(df.groupby(['targets']).mean()['score'], df.groupby(['targets']).mean()['score'].mean())
# print(df)
# print(targets, y_pred_tags)