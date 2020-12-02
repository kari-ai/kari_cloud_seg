import random
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageAug:
    def __init__(self, hyp=None):
        self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                             A.VerticalFlip(p=0.5),
                             A.ShiftScaleRotate(p=0.5),
                             A.RandomBrightnessContrast(p=0.3),
                             ToTensorV2()])

    def __call__(self, img, label):
        img = img.permute(1, 2, 0).numpy().astype(dtype=np.float32) / (2 ** 14 - 1)
        label = label.numpy()
        transformed = self.aug(image=img, mask=label)
        return transformed['image'], transformed['mask']


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            pass
            # img = TF.hflip(img)
            # target = TF.hflip(target)
        return img, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            pass
            # img = TF.vflip(img)
            # target = TF.vflip(target)
        return img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target=None):
        mean = torch.as_tensor(self.mean)
        std = torch.as_tensor(self.std)
        if (std == 0).any():
            raise ValueError('some std is zero')
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        img = img.sub(mean).div(std)
        target = None
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    def __call__(self, img, target=None):
        if img is not None:
            img = torch.from_numpy(img.transpose(2, 0, 1)).to(dtype=torch.int16)  # (H, W, C) to (C, H, W)
        else:
            img = None

        if target is not None:
            target = torch.from_numpy(target).to(dtype=torch.uint8)  # tensor of (H, W) belongs to {0, 1, 2, 3}
        else:
            target = None

        return img, target


def to_tensor():
    transforms = [ToTensor()]
    return Compose(transforms)


def get_transforms():
    transforms = ImageAug()
    return transforms
