import logging
import random
import os
from osgeo import gdal
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
from pathlib import Path
import glob
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def set_logging(level='logging.INFO'):
    logging.basicConfig(format="%(message)s", level=level)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def increment_dir(dir, comment=''):
    # Increments a directory runs/run0 --> runs/run1_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    # d = sorted(glob.glob(dir + '*'))  # directories
    # if len(d):
    #    n = max([int(x[len(dir):x.find('run_') if '_' in Path(x).name else None]) for x in d]) + 1  # increment
    # return dir + str(n) + ('_' + comment if comment else '')
    return dir + ('_' + comment if comment else '')


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availability

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        logger.info('[GPU Mode]')
        for i in range(0, ng):
            logger.info("  Device %g: %s (total_memory=%dMB)" % (i, x[i].name, x[i].total_memory / c))
        torch.backends.cudnn.benchmark = True   # for faster training
    else:
        logger.info('[CPU Mode]')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kari_geotiff_read(filename, num_bands=4):
    ds = gdal.Open(filename)
    band = []
    for i in range(num_bands):
        band.append(ds.GetRasterBand(i + 1).ReadAsArray())  # ranges 0-255
    img = np.dstack(band).astype(np.int16)
    return img


def cv2_imshow(img, label=None, window_name='img', max_window_size=(4096, 2160), display=True, write_path=None):
    # img
    if torch.is_tensor(img):
        # tensor in [0,1] to numpy array in [0, 255]
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = img[:, :, :3] * 255

    else:
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3] / (2 ** 14 - 1) * 255

    img = img[..., ::-1].astype(np.uint8)
    out_img = img
    h, w = out_img.shape[:2]

    # label
    if label is not None:
        if torch.is_tensor(label):
            # tensor in [0, 1, 2, 3] to color image
            label_color = np.zeros((h, w, 3), dtype=np.uint8)
            label_np = label.squeeze().cpu().numpy()
            pos = label_np == 1
            label_color[pos] = [0, 0, 255]
            pos = label_np == 2
            label_color[pos] = [0, 255, 0]
            pos = label_np == 3
            label_color[pos] = [0, 255, 255]
        else:   # numpy array
            if label.ndim == 3 and label.shape[2] == 3:   # color numpy array
                label_color = label
            elif label.ndim == 2:                         # numpy array, but 2d in [0, 1, 2, 3]
                label_color = np.zeros((h, w, 3), dtype=np.uint8)
                pos = label == 1
                label_color[pos] = [0, 0, 255]
                pos = label == 2
                label_color[pos] = [0, 255, 0]
                pos = label == 3
                label_color[pos] = [0, 255, 255]

        out_img = cv2.addWeighted(label_color, 0.3, out_img, 1, 0)

    # display
    max_h, max_w = max_window_size
    h_ratio, w_ratio = h / max_h if h / max_h > 1.0 else 1.0, w / max_w if w / max_w > 1.0 else 1.0
    ratio = max(h_ratio, w_ratio)
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(h * ratio), int(w * ratio))
        cv2.imshow(window_name, out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if write_path is not None:
        out_filename = os.path.join(write_path, window_name + '.png')
        cv2.imwrite(out_filename, out_img)


def cv2_predshow(pred, img=None, window_name='img', max_window_size=(4096, 2160)):
    # pred
    assert pred.shape[0] == 4, 'dimension must be 4 (four classes)'
    pred_img = np.zeros((*list(pred.shape[1:]), 3), dtype=np.uint8)
    _, idx = torch.max(pred, 0)
    pos = idx == 0
    pred_img[pos.cpu().numpy()] = [0, 0, 0]
    pos = idx == 1
    pred_img[pos.cpu().numpy()] = [0, 0, 255]
    pos = idx == 2
    pred_img[pos.cpu().numpy()] = [0, 255, 0]
    pos = idx == 3
    pred_img[pos.cpu().numpy()] = [0, 255, 255]
    h, w = pred_img.shape[:2]

    # img
    if img is not None:
        # tensor in [0,1] to numpy array in [0, 255]
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = img[:, :, :3] * 255
        img = img[..., ::-1].astype(np.uint8)
        out_img = img
        out_img = cv2.addWeighted(pred_img, 0.3, out_img, 1, 0)
    else:
        out_img = pred_img

    # display
    max_h, max_w = max_window_size
    h_ratio, w_ratio = h / max_h if h / max_h > 1.0 else 1.0, w / max_w if w / max_w > 1.0 else 1.0
    ratio = max(h_ratio, w_ratio)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(h * ratio), int(w * ratio))
    cv2.imshow(window_name, out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def moving_avg(x, n):
    l = len(x)
    p = min(l, n)
    return x[-p:].sum()/float(p)


def collate_fn(batch):
    return tuple(zip(*batch))


def fitness_test(true, pred, num_classes=4):
    eps = 1e-7
    true_one_hot = F.one_hot(true.squeeze(1), num_classes=num_classes)  # (B, 1, H, W) to (B, H, W, C)
    true_one_hot = true_one_hot.permute(0, 3, 1, 2)  # (B, H, W, C) to (B, C, H, W)
    pred_max = pred.argmax(1)      # (B, C, H, W) to (B, H, W)
    pix_acc = (true == pred_max.unsqueeze(1)).sum().float().div(true.nelement())
    pred_one_hot = F.one_hot(pred_max, num_classes=num_classes)   # (B, H, W) to (B, H, W, C)
    pred_one_hot = pred_one_hot.permute(0, 3, 1, 2)   # (B, H, W, C) to (B, C, H, W)

    true_one_hot = true_one_hot.type(pred_one_hot.type())
    dims = (0,) + tuple(range(2, true.ndimension()))  # dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot & true_one_hot, dims)
    union = torch.sum(pred_one_hot | true_one_hot, dims)
    m_iou = (intersection / (union + eps)).mean()

    return m_iou.item(), pix_acc.item()
