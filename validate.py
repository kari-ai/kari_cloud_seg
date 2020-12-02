import os
import argparse
import logging
import torch
from utils.utils import set_logging, init_seeds, select_device, cv2_imshow, increment_dir, fitness_test, count_parameters
from data.kari_cloud_dataset import KariCloudDataset
import yaml
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from models.my_dilated_conv_unet import MyDilatedConvUNet
from models.hrnet import get_seg_model
from models.config import update_config
from tqdm import tqdm
import numpy as np
from loss import dice_loss, jaccard_loss, ce_loss
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


def validate(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyper-parameters {hyp}')
    # log_dir = Path(tb_writer.log_dir)
    rank = opt.global_rank

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_path = data_dict['train']
    val_path = data_dict['val']
    num_classes, class_names = (data_dict['num_classes'], data_dict['class_names'])
    assert num_classes == len(class_names), '%g names found for nc=%g dataset in %s' % \
                                            (len(class_names), num_classes, opt.data)
    cache_path = opt.cache_dir
    os.makedirs(cache_path, exist_ok=True)
    patch_size = opt.patch_size
    patch_stride = opt.patch_stride
    batch_size = opt.batch_size

    # Train dataset
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 64])  # number of workers

    # Validation dataset
    val_dataset = KariCloudDataset(val_path, cache_path, patch_size, patch_stride, is_train=False, load_label=True,
                                   mean=None, std=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=nw,
                                                 pin_memory=True, drop_last=True)

    weight_file = opt.weights

    num_epochs = opt.epochs

    if os.path.exists(weight_file):
        checkpoint = torch.load(weight_file)
        opt.model = checkpoint['model_name']
        epoch = checkpoint['epoch'] + 1
        best_fit = checkpoint['best_fit']
        print("Epoch %g..." % epoch)

    if opt.model == 'deeplabv3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=4)
        model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif opt.model == 'hrnet_w18':
        hrnet_cfg = update_config('models/hrnet_w18_config.yaml')
        model = get_seg_model(hrnet_cfg)
    elif opt.model == 'hrnet_w48':
        hrnet_cfg = update_config('models/hrnet_w48_config.yaml')
        model = get_seg_model(hrnet_cfg)
    elif opt.model == 'dilated_unet':
        model = MyDilatedConvUNet()
    else:
        logger.critial('unsupported model')
        exit(1)

    print('number of parameters: ', count_parameters(model))


    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)


    model.load_state_dict(checkpoint['model'])


    # Start training


    # validation
    fit = val_one_epoch(model, val_dataloader, device, epoch, num_epochs, patch_size, tb_writer)


def val_one_epoch(model, data_loader, device, epoch, num_epochs, patch_size, tb_writer):
    model.eval()
    m_iou_list = np.array([])
    pix_acc_list = np.array([])

    logger.info(('\n' + '%10s' * 3) % ('Epoch(V)', 'mIOU', 'Accuracy'))
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    s = ('%10s' + '%10.4f' + ' %8.4f') % (
        '-' % 0.0, 0.0)
    pbar.set_description(s)

    for i, (imgs, targets) in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            if opt.model == 'deeplabv3':
                preds = model(imgs)['out']
                targets = targets.long()
            elif opt.model == 'hrnet_w18' or opt.model == 'hrnet_w48':
                preds = model(imgs)
                h, w = preds.shape[2], preds.shape[3]
                targets = F.interpolate(targets.float(), size=(h, w), mode='nearest').long()
            elif opt.model == 'dilated_unet':
                preds = model(imgs)
                targets = targets.long()

            m_iou, pix_acc = fitness_test(targets, preds)

            s = ('%10s' + '%10.4f' + ' %8.4f') % (
                '%g' % epoch, m_iou, pix_acc)
            pbar.set_description(s)
            m_iou_list = np.append(m_iou_list, m_iou)
            pix_acc_list = np.append(pix_acc_list, pix_acc)
    val_m_iou_mean = m_iou_list.mean()
    val_pix_acc_mean = pix_acc_list.mean()
    logger.info('[V] mIOU={:.3f}, Accuracy={:.3f}'.format(val_m_iou_mean, val_pix_acc_mean))
    tb_writer.add_scalar('val_epoch_m_iou', val_m_iou_mean, epoch)
    tb_writer.add_scalar('val_epoch_pix_acc', val_pix_acc_mean, epoch)
    return val_pix_acc_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/kari_cloud.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--cache-dir', type=str, default='data/caches', help='cache path')
    parser.add_argument('--model', type=str, default='deeplabv3', help='models: deeplabv3, dilated_unet, '
                                                                       'hr_w18, or hr_w48')
    parser.add_argument('--loss', type=str, default='dice', help='losses: ce, dice, or jaccard')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--patch-size', type=int, default=800, help='patch sizes')
    parser.add_argument('--patch-stride', type=int, default=400, help='patch sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--weights', type=str, default='weights/deeplabv3_best.pt', help='model.pt path')
    parser.add_argument('--name', default='kari_seg', help='name for the run')
    opt = parser.parse_args()

    opt.global_rank = -1

    set_logging(level=logging.INFO)
    logger.info(opt)

    # Load hyper-parameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Tensorboard
    log_dir = increment_dir(Path(opt.logdir) / 'run', opt.name)
    tb_writer = SummaryWriter(log_dir=log_dir)

    # Train
    validate(hyp, opt, device, tb_writer)
