import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from models.my_dilated_conv_unet import MyDilatedConvUNet
from models.hrnet import get_seg_model
from models.config import update_config
from data.kari_cloud_transforms import to_tensor
from data.kari_cloud_dataset import KariCloudDataset
from utils.utils import kari_geotiff_read, cv2_imshow, cv2_predshow
import argparse
import yaml
from torchvision import models


def predict(opt):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = MyTernausNet()
    # model = MyDilatedConvUNet()
    hrnet_cfg = update_config('models/hrnet_w18_config.yaml')
    # model = get_seg_model(hrnet_cfg)
    model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=4)
    model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # print(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    assert os.path.exists(opt.weights), "no found model weights"
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['model'])

    patch_size = opt.patch_size
    stride = opt.patch_stride
    batch_size = opt.batch_size


    transform = to_tensor()
    # Input
    for test_file in opt.input:

        win_name = os.path.basename(test_file).rsplit('.tif')[0]

        if opt.val:
            label_file = test_file.replace('img' + os.sep + 'tif', 'label').replace('.tif', '_label.png')
            label_img = cv2.imread(label_file)

        model.eval()
        img = kari_geotiff_read(test_file)
        h, w = img.shape[:2]
        img, _ = transform(img, None)        # int16
        img = img.float().div(2 ** 14 - 1)   # float32 [0, 1]
        # unfold
        pad_h = int((np.ceil(h / stride) - 1) * stride + patch_size - h)
        pad_w = int((np.ceil(w / stride) - 1) * stride + patch_size - w)
        padded_img = F.pad(img, pad=[0, pad_w, 0, pad_h])
        patches = padded_img.unfold(1, patch_size, stride).unfold(2, patch_size,
                                                                        stride)  # [C, NH, NW, H, W]
        pred_patches = []
        for y in range(patches.shape[1]):
            imgs = patches[:, y, :, :, :].permute(1, 0, 2, 3).to(device)
            with torch.no_grad():
                preds = model(imgs)['out']
                preds = F.interpolate(preds, size=patch_size, mode='bilinear', align_corners=False)
                pred_patches += preds.unsqueeze(0).cpu()
                del preds
            # for x in range(preds.shape[0]):
            #     cv2_predshow(preds[x], imgs[x])
        pred_patches = torch.stack(pred_patches).permute(2, 0, 1, 3, 4).unsqueeze(0)
        pred_patches = pred_patches.contiguous().view(1, 4, -1, patch_size * patch_size)
        pred_patches = pred_patches.permute(0, 1, 3, 2)
        pred_patches = pred_patches.contiguous().view(1, 4 * patch_size * patch_size, -1)
        ph, pw = padded_img.shape[1], padded_img.shape[2]
        out = F.fold(pred_patches, output_size=(ph, pw), kernel_size=patch_size, stride=stride)  # [B, C, H, W]
        recovery_mask = F.fold(torch.ones_like(pred_patches), output_size=(ph, pw), kernel_size=patch_size, stride=stride)
        out /= recovery_mask

        pred_img = np.zeros((*list(out.shape[2:]), 3), dtype=np.uint8)
        _, idx = out.squeeze(0).max(0)
        pos = idx == 0
        pred_img[pos.numpy()] = [0, 0, 0]
        pos = idx == 1
        pred_img[pos.cpu().numpy()] = [0, 0, 255]
        pos = idx == 2
        pred_img[pos.cpu().numpy()] = [0, 255, 0]
        pos = idx == 3
        pred_img[pos.cpu().numpy()] = [0, 255, 255]

        cv2_imshow(img, pred_img[:h, :w, :], win_name + '_pred', close=False, write_path='./outputs/')
        if opt.val:
            acc = np.all(pred_img[:h, :w, :] == label_img, axis=2).sum()/(w * h)
            print('accuracy=', acc)

            cv2_imshow(img, label_img, win_name + '_target', write_path='./outputs/')

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

        # for y in range(patches.shape[1]):
        #     for x in range(patches.shape[2]):
        #         cv2_imshow(patches[:, y, x, :, :])




    # Train dataset

    # model.eval()
    # acc_sum = 0
    # num_imgs_sum = 0
    # for i, (imgs, targets) in enumerate(test_dataloader):
    #     imgs, targets = imgs.to(device), targets.to(device)
    #     preds = model(imgs)
    #     _, p = preds.max(1)
    #     acc_sum += (targets == p).sum().float().cpu() / (patch_size * patch_size)
    #     num_imgs_sum += len(imgs)
    #     print(acc_sum/num_imgs_sum)
    #     #for j in range(preds.shape[0]):
    #     #    cv2_predshow(preds[j], imgs[j])
    # print(acc_sum, num_imgs_sum)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--weights', type=str, default='weights/deeplabv3_best.pt', help='model.pt path')
    parser.add_argument('--cache-dir', type=str, default='data/caches', help='cache path')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--patch-size', type=int, default=800, help='patch sizes')
    parser.add_argument('--patch-stride', type=int, default=200, help='patch sizes')
    parser.add_argument('--val', '-v', type=bool, help="Validate the performance", default=True)
    opt = parser.parse_args()
    print(opt)

    predict(opt)
