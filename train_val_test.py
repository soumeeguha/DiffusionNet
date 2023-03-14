from base64 import encode
import torch
from tqdm import tqdm
import torch.nn as nn
from typing import Union



def multi_acc(y_pred, y_test, test = False):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)        
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)    
    acc = torch.round(acc * 100)  
    if test:
        return acc, y_pred_tags
    else:  
        return acc

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


def train(epoch, dataloader, model, device, criterionC, optimizerC, schedulerC):
    
    model.train()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.7)  # progress bar enumeration
    loss_batch = []
    train_acc = []
    
    for batch_num, (x, img, y_tar) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)

        x, img, y_tar = ship_device([x, img, y_tar], device)


        model.zero_grad()
        y_out = model(x.float(), img)
        class_loss = criterionC(y_out, y_tar)

        class_loss.backward()
        class_loss = class_loss.item()
        optimizerC.step()

        tqdm_enum.set_description(f"Train  E: {epoch} - BCE: {class_loss:.3}  Acc: {multi_acc(y_out[1], y_tar)}")

    schedulerC.step()

def val(epoch, dataloader, model, device, criterionC):
    model.eval()
    # encoder.eval()
    
    with torch.no_grad():
        
        tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.7)  # progress bar enumeration
        loss_batch = []

        
        for batch_num, (x, img, y_tar) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)

            x, img, y_tar = ship_device([x, img, y_tar], device)
            
            y_out = model(x.float(), img)
            class_loss = criterionC(y_out, y_tar)
            class_loss = class_loss.item()

            tqdm_enum.set_description(f"Val  E: {epoch} - BCE: {class_loss:.3}  Acc: {multi_acc(y_out[1], y_tar)}")

def test(epoch, dataloader, model, device, criterionC):
    model.eval()
    
    with torch.no_grad():
        
        tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.7)  # progress bar enumeration
        loss_batch = []

        ground_truth = torch.tensor((0, 0))
        predictions = torch.tensor((0, 0))
        diffs = torch.tensor((0, 0))

        for batch_num, (x, img, y_tar) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)

            x, img, y_tar = ship_device([x, img, y_tar], device)
            
            y_out = model(x.float(), img)
            class_loss = criterionC(y_out, y_tar)
            class_loss = class_loss.item()


            y_tar, pred_lbls = y_tar.detach().cpu(), y_out[1].detach().cpu()

            if batch_num == 0:
                targets = y_tar
                preds = pred_lbls
            else: 
                targets = torch.cat((targets, y_tar), 0)
                preds = torch.cat((preds, pred_lbls), 0)


        print(targets.size(), preds.size())
        acc, y_pred_tags = multi_acc(preds, targets, test = True)

    return targets, preds, acc, y_pred_tags

def test_real_imgs(dataloader, model, device, criterionC):
    model.eval()
    
    with torch.no_grad():
        
        tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.7)  # progress bar enumeration
        loss_batch = []

        ground_truth = torch.tensor((0, 0))
        predictions = torch.tensor((0, 0))
        diffs = torch.tensor((0, 0))

        for batch_num, (x, img, y_tar) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)

            x, img, y_tar = ship_device([x, img, y_tar], device)
            
            y_out = model(x.float(), img)
            # class_loss = criterionC(y_out, y_tar)
            # class_loss = class_loss.item()


            y_tar, pred_lbls = y_tar.detach().cpu(), y_out[1].detach().cpu()
            m = nn.Softmax(dim=1)
            predicted_probs = m(pred_lbls)

            y_pred_softmax = torch.log_softmax(pred_lbls, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            # print(y_pred_tags)

            # if batch_num == 0:
            #     targets = y_tar
            #     preds = pred_lbls
            # else: 
            #     targets = torch.cat((targets, y_tar), 0)
            #     preds = torch.cat((preds, pred_lbls), 0)


        # print(targets.size(), preds.size())
        # acc, y_pred_tags = multi_acc(preds, targets, test = True)

    return y_pred_tags, y_tar, predicted_probs