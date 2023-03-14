import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Union

def ship_device(x, device: Union[str, torch.device]):
    """
    Ships the input to a pytorch compatible device (e.g. CUDA)

    Args:
        x:
        device:

    Returns:
        x

    """
    if x is None:
        return x

    elif isinstance(x, torch.Tensor):
        return x.to(device)

    elif isinstance(x, (tuple, list)):
        x = [ship_device(x_el, device) for x_el in x]  # a nice little recursion that worked at the first try
        return x

    elif device != 'cpu':
        raise NotImplementedError(f"Unsupported data type for shipping from host to CUDA device.")



def train_ae(epoch, dataloader, model, encoder, device, criterionE, optimizerE, schedulerE):
    
    model.train()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.7)  # progress bar enumeration
    loss_batch = []
    train_acc = []
    
    for batch_num, (x, img, y_tar) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)

        x, img, y_tar = ship_device([x, img, y_tar], device)

        encoder.zero_grad()
        # print(img.max(), img.min())
        encoded, img_reconstructed = encoder(img)
        # print(img_reconstructed.max(), img_reconstructed.min())
        # print(encoded.size(), img_reconstructed.size(), img.size())
        enc_loss = 10*criterionE(img_reconstructed, img).mean()
        optimizerE.zero_grad()
        
        
        if (torch.any(encoded.isnan())):
            print('encoded: ', encoded)
            break

        tqdm_enum.set_description(f"Train  E: {epoch} -MSE: {enc_loss:.3}")

    schedulerE.step()

    # print('epoch: ', epoch, 'encoded: ', encoded, 'recondtructed: ', img_reconstructed)