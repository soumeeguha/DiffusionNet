import torch
import torch.distributions as distributions
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def update_image(k, image, mean, cov):
    sampler = distributions.multivariate_normal.MultivariateNormal(mean, cov)
    image[mean] = 1
    pts = sampler.sample((k,))
    s = pts.round()
    for i in range(s.size()[0]):
        indices = s.select(0, i)
        image[int(indices[0].item()), int(indices[1].item())] += 0.5
    return image

def get_image(tar_img, mean1, mean2, cov1, cov2):
    k = 1000
    image = torch.zeros_like(tar_img)
    image = update_image(k, image, mean1, cov1)
    image = update_image(k, image, mean2, cov2)
    return image