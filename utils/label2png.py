# -*- coding: utf-8 -*-

import os.path as osp
import shutil
import os
from tqdm import tqdm
import cv2


def tif2png():
    # folder_path = osp.join(sys_path, "ISBI2016_ISIC_Part1_Training_GroundTruth")
    # save_folder = osp.join(sys_path, "labels")
    folder_path =  'COVID/train_gt'
    save_folder =  'COVID/trainLabel'
    if osp.isdir(save_folder):
        # remove
        shutil.rmtree(save_folder)
        # new
        os.makedirs(save_folder)
    else:
        # print(save_folder)
        os.makedirs(save_folder)

    for name in os.listdir(folder_path):
        print(name)
    images = os.listdir(folder_path)
    with tqdm(total=len(images)) as pbar:
        for image in images:
            image_name = image.split("_S")[0]
            src_path = osp.join(folder_path, image)
            img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            img[img == 255] = 1
            save_path = osp.join(save_folder, image_name + ".png")
            cv2.imwrite(save_path, img)
            pbar.update(1)
    print("label convert done")


if __name__ == '__main__':
    tif2png()
