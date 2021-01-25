import os
from utils.utils import kari_geotiff_read, cv2_imshow


if __name__ == '__main__':
    img_files = ['data/kari_cloud/img/tif/CLD00010_MS4_K3A_NIA0010.tif',
                   'data/kari_cloud/img/tif/CLD00015_MS4_K3A_NIA0015.tif',
                   'data/kari_cloud/img/tif/CLD00064_MS4_K3_NIA0064.tif',
                   'data/kari_cloud/img/tif/CLD00072_MS4_K3_NIA0072.tif',
                   'data/kari_cloud/img/tif/CLD00184_MS4_K3A_NIA0498.tif',
                   'data/kari_cloud/img/tif/CLD00215_MS4_K3A_NIA1062.tif']
    for img_file in img_files:
        win_name = os.path.basename(img_file).rsplit('.tif')[0]
        img = kari_geotiff_read(img_file)
        cv2_imshow(img, label_img=None, window_name=win_name, write_path='./outputs/')
