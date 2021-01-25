import numpy as np
import glob
import cv2
import random
from utils.utils import kari_geotiff_read, cv2_imshow


if __name__ == '__main__':
    label_files = glob.glob('./label/*.png')
    random.shuffle(label_files)
    for label_file in label_files:
        print(label_file)
        label = cv2.imread(label_file)
        img_file = label_file.replace('./label', './img/tif').replace('_label.png', '.tif')
        img = kari_geotiff_read(img_file)
        cv2_imshow(img, label)
