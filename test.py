# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
import os

import numpy
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results
from utils.utils_metrics import Dice_loss,Focal_Loss,f1_lossFromTensor
from utils.DiceLoss import SoftDiceLoss
import glob
from utils.plot import plot_confusion_matrix
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet,Unet_att_ECA,Unet_CBAM
from model.unet_cbam import UnetCBAM
from PIL import Image
from train import  getArgs

def cal_miou(net,model_path,test_dir="COVID/test",
             pred_dir='result_att/0-1mask', gt_dir="COVID/test_gt/0-1mask",
             miou_out_path="result_att/miou"
             ):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["background", "infection"]
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    # 计算结果和gt的结果进行比对
    result_dir = os.path.split(pred_dir)[0]

    GT_dir = os.path.join(result_dir,'0-255mask')
    GT_contrast  = os.path.join(result_dir,'0-255contrast')
    label_path = os.path.split(test_dir)[0]
    label_path = os.path.join(label_path,'test_gt','0-255mask')

    if not os.path.exists(GT_dir):
        os.makedirs(GT_dir)
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)
    if not os.path.exists(GT_contrast):
        os.makedirs(GT_contrast)
    # 加载模型
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。

        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load(model_path, map_location=device))
        # 测试模式
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".png")

            #
            img = cv2.imread(image_path)
            origin_shape = img.shape
            # print(origin_shape)
            # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (256, 256))
            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            pred = net(img_tensor)
            # 提取结果
            # img_name = image_id + ".png"
            # target_name = os.path.join('COVID/test_gt/0-1mask',img_name)
            # target = cv2.imread(target_name)
            # target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            # target = target.reshape(1, 1, target.shape[0], target.shape[1])
            # target = torch.from_numpy(target)
            # target = target.to(device=device, dtype=torch.float32)
            # F_score = f1_lossFromTensor(target,pred)
            # print(f'======================>f_score:{F_score}')
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision,f_score,plot_hist = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        confusion_matrix = get_confusion_matrix(hist)
        class_name = ['infection','background']
        martix_path = os.path.join(miou_out_path,'martix.png')
        plot_confusion_matrix(confusion_matrix,class_name,martix_path)

        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision,f_score, name_classes)
        getGT(pred_dir,GT_dir)
        get_gt_contrast(pred_dir,GT_contrast,label_dir=label_path)

def get_confusion_matrix(hist:np.ndarray):
    hist = hist[[1, 0], :]
    hist = hist[:, [1, 0]]
    return hist

def getGT(pre_dir,GT_dir):
    for name  in os.listdir(pre_dir):
        if name.endswith('.png'):
            img_name = os.path.join(pre_dir,name)
            img = Image.open(img_name)
            arr = numpy.array(img)
            arr[arr==1] = 255
            img = Image.fromarray(arr)
            img.save(os.path.join(GT_dir,name))
def get_gt_contrast(pre_dir,GT_dir,label_dir):
    for name in os.listdir(pre_dir):
        if name.endswith('.png'):
            img_name = os.path.join(pre_dir, name)
            img = Image.open(img_name)
            arr = numpy.array(img)
            arr[arr == 1] = 255
            lable_name = os.path.join(label_dir,name)
            label = Image.open(lable_name)
            label = np.array(label)
            mask = np.column_stack((arr,label))
            img = Image.fromarray(mask)
            img.save(os.path.join(GT_dir, name))
if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1)
    arg = getArgs()
    pred_dir = os.path.join(f'result_{arg.model}_{arg.end_epoch}','0-1mask')
    miou_path = os.path.join(f'result_{arg.model}_{arg.end_epoch}','miou')
    cal_miou(net = net,model_path='logs/Unet_50/Unet_model_49.pth',pred_dir=pred_dir,miou_out_path=miou_path)

