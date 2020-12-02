import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
# from osgeo import gdal
import cv2
from data.kari_cloud_transforms import get_transforms, to_tensor
from utils.utils import kari_geotiff_read
# import torch.utils.data
from pathlib import Path
from tqdm import tqdm


class KariCloudDataset(torch.utils.data.Dataset):
    def __init__(self, path, cache_path, patch_size=800, patch_stride=400, is_train=False, load_label=False,
                 mean=None, std=None):
        if is_train:
            load_label = True
        try:
            img_files = []
            label_files = []
            p = str(Path(path))
            if os.path.isfile(p):  # list file
                with open(p, 'r') as t:
                    t = t.read().splitlines()
                    img_files += [x for x in t]
                    if load_label:
                        label_files += [x.replace('img/tif', 'label').replace('.tif', '_label.png')
                                        for x in t]
            elif os.path.isdir(p):
                img_files += glob.iglob(p + os.sep + "*.*")
            else:
                raise Exception('%s does not exit' % p)
        except Exception as e:
            raise Exception('Error loading data from %s: %s' % (path, e))

        self.cache_path = cache_path
        self.is_train = is_train
        self.load_label = load_label
        self.img_files = img_files
        self.label_files = label_files
        self.to_tensor = to_tensor()
        self.transforms = get_transforms()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.img_patch_files = []
        self.label_patch_files = []

        pbar = tqdm(enumerate(self.img_files), total=len(self.img_files))
        # check cache
        for i, img_file in pbar:
            # print(img_file)
            img_cache_file = os.path.join(cache_path,
                                          'img_%g_%g_%s_0_0.pt' % (patch_size, patch_stride,
                                                                   os.path.basename(img_file).rsplit('.tif')[0]))
            if not os.path.isfile(img_cache_file):
                self.create_img_cache(img_file)
                desc_str = 'Creating'
                desc_str2 = 'created'
            else:
                desc_str = 'Scanning'
                desc_str2 = 'found'
            self.img_patch_files += glob.glob(img_cache_file.replace('0_0.pt', '*.pt'))

            pbar.desc = '%s image patches in ./%s... (%g %s)' % (desc_str, cache_path, len(self.img_patch_files),
                                                                 desc_str2)

        if load_label:
            pbar = tqdm(enumerate(self.label_files), total=len(self.label_files))
            for i, label_file in pbar:
                label_cache_file = os.path.join(cache_path,
                                                'label_%g_%g_%s_0_0.pt' % (patch_size, patch_stride,
                                                 os.path.basename(label_file).rsplit('_label.png')[0]))
                if not os.path.isfile(label_cache_file):
                    self.create_label_cache(label_file)
                    desc_str = 'Creating'
                    desc_str2 = 'created'
                else:
                    desc_str = 'Scanning'
                    desc_str2 = 'found'
                self.label_patch_files += glob.glob(label_cache_file.replace('0_0.pt', '*.pt'))
                pbar.desc = '%s label patches in ./%s... (%g %s)' % (desc_str, cache_path, len(self.label_patch_files),
                                                                     desc_str2)

    def create_img_cache(self, img_file):
        img = kari_geotiff_read(img_file)
        h, w = img.shape[:2]

        # numpy arrays to tensors
        target = None
        img, target = self.to_tensor(img, target)

        patch_size = self.patch_size
        patch_stride = self.patch_stride
        pad_h = int((np.ceil(h / patch_stride) - 1) * patch_stride + patch_size - h)
        pad_w = int((np.ceil(w / patch_stride) - 1) * patch_stride + patch_size - w)
        padded_img = F.pad(img, pad=[0, pad_w, 0, pad_h])
        patches = padded_img.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride) # [C, NH, NW, H, W]

        for y in range(patches.shape[1]):
            for x in range(patches.shape[2]):
                # from utils.utils import cv2_imshow
                # cv2_imshow(patches[:, y, x, :, :])
                cache_img_file = os.path.join(self.cache_path, 'img_%g_%g_%s_%g_%g.pt' %
                                              (self.patch_size, self.patch_stride,
                                               os.path.basename(img_file).rsplit('.tif')[0], y, x))
                torch.save(patches[:, y, x, :, :].contiguous(), cache_img_file)

        # fold
        # patches = patches.unsqueeze(0)              # [B, C, NH, NW, H, W], here B=1
        # patches = patches.contiguous().view(1, 4, -1, patch_size*patch_size)  # [B, C, NH * NW, patch_size*patch_size]
        # patches = patches.permute(0, 1, 3, 2)       # [B, C, patch_size * patch_size, NH * NW]
        # patches = patches.contiguous().view(1, 4*patch_size*patch_size, -1)   # [B, C*patch_size*patch_size, L]
        # ph, pw = padded_img.shape[1], padded_img.shape[2]
        # out = F.fold(patches, output_size=(ph, pw), kernel_size=patch_size, stride=stride)  # [B, C, H, W]
        # recovery_mask = F.fold(torch.ones_like(patches), output_size=(ph, pw), kernel_size=patch_size, stride=stride)
        # out /= recovery_mask                       # normalization to prevent the overlapping area from being added
        # cv2_imshow(out.squeeze())

        # for i in range(patches.shape[1]):
        #     for j in range(patches.shape[2]):
        #         print(i, j)
        #         cv2_imshow(patches[:,i,j,:,:])

    def create_label_cache(self, label_file):
        label = cv2.imread(label_file)
        h, w = label.shape[:2]

        # numpy arrays to tensors
        target = np.zeros((h, w), dtype=np.uint8)
        pos = np.where(np.all(label == [0, 0, 255], axis=-1))  # thick cloud
        target[pos] = 1
        pos = np.where(np.all(label == [0, 255, 0], axis=-1))  # thin cloud
        target[pos] = 2
        pos = np.where(np.all(label == [0, 255, 255], axis=-1))  # cloud shadow
        target[pos] = 3
        img, target = self.to_tensor(None, target)

        patch_size = self.patch_size
        patch_stride = self.patch_stride
        pad_h = int((np.ceil(h / patch_stride) - 1) * patch_stride + patch_size - h)
        pad_w = int((np.ceil(w / patch_stride) - 1) * patch_stride + patch_size - w)
        padded_target = F.pad(target, pad=[0, pad_w, 0, pad_h])
        patches = padded_target.unfold(0, patch_size, patch_stride).unfold(1, patch_size, patch_stride) # [NH, NW, H, W]


        for y in range(patches.shape[0]):
            for x in range(patches.shape[1]):
                # from utils.utils import cv2_imshow
                # cv2_imshow(patches[y, x, :, :])
                cache_label_file = os.path.join(self.cache_path, 'label_%g_%g_%s_%g_%g.pt' %
                                              (self.patch_size, self.patch_stride,
                                               os.path.basename(label_file).rsplit('_label.png')[0], y, x))
                torch.save(patches[y, x, :, :].contiguous(), cache_label_file)

    def __getitem__(self, idx):
        img_patch_file = self.img_patch_files[idx]
        img = torch.load(img_patch_file)
        if self.load_label:
            label_patch_file = img_patch_file.replace('img_', 'label_')
            target = torch.load(label_patch_file)  # [H, W]
        else:
            target = None
        if self.is_train:
            img, target = self.transforms(img, target)
        else:
            img = img.float().div(2 ** 14 - 1)
        return img, target.unsqueeze(0)  # target [1, H, W]

    def __len__(self):
        return len(self.img_patch_files)

