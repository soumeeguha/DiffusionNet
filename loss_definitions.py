import torch
import torch.nn as nn
# import LabelSmoothingCrossEntropy

import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.softmax = nn.Softmax(dim = 1)

    def forward(self,y_pred, y_true, epsilon = 0.00000001):
        # print(y_true, y_pred, y_pred[:, 0])
        y_pred = self.softmax(y_pred) + epsilon
        y_pred = torch.clip(y_pred, min = 1e-7, max = 1 - 1e-7)
        term_0 = (1-y_true) * torch.log(y_pred[:,0] + 1e-7)
        term_1 = y_true * torch.log(y_pred[:, 1] + 1e-7)
        return -torch.mean(term_0+term_1, axis=0)

    #     def BinaryCrossEntropy(y_true, y_pred):
    # y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    # term_1 = y_true * np.log(y_pred + 1e-7)
    # return -np.mean(term_0+term_1, axis=0)



def BCE_MSE(y_tar, y_out, img, img_reconstructed):
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()

    loss = 2*criterion1(y_out, y_tar) + criterion2(torch.flatten(img, start_dim = 1), img_reconstructed)
    return loss

class BCE_MSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.MSELoss()

    def forward(self, y_out, y_tar, img, img_reconstructed):
        mse = self.criterion2(img, img_reconstructed)
        # print(y_out.size(), y_tar.size())
        bce = 2*self.criterion1(y_out, y_tar)
        return bce, mse

class MaxEntropyFGC(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim = 1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, pred, target, gamma = 1000, epsilon = 0.000001):
        p = self.softmax(pred) + epsilon
        entropy = torch.mean(p*torch.log(p))
        y = nn.functional.one_hot(target)
        kl_div = self.kl_loss(y, p)

        return kl_div - gamma*entropy

class MaxEntropyAttnMaps(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim = 1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        # self.ce = nn.CrossEntropyLoss()
        self.ce = LabelSmoothingCrossEntropy(reduction='sum')

    def forward(self, out, target, gamma = 0.1, epsilon = 0.000001):
        
        attn, pred = out[0].flatten(start_dim = 1), out[1]
        attn = self.softmax(attn) + epsilon
        entropy = torch.mean(attn*torch.log(attn))
        p = self.softmax(pred) + epsilon
        # print(torch.round(target.float()).int())
        # print(target, target.dtype)

        y = nn.functional.one_hot(torch.round(target.float()).int().long())
        # y = nn.functional.one_hot(target)
        # print('one hot shape: ', y.size())
        # y = target #already binary
        # print('without one hot shape: ', y.size())
        ce = self.ce(pred, target)

        return ce - gamma*entropy
        # return ce