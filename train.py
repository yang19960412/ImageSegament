import argparse
import os

import numpy as np
from model.unet_model import Unet_att_ECA, UNet, Unet_CBAM
from model.unet_cbam import UnetCBAM
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from utils.log import LossHistory
from utils.utils_metrics import dice_coeff, f1_lossFromTensor, getF1_ScoreByNumpy
from utils.DiceLoss import SoftDiceLoss, BCEDiceLoss
from model.CooAttPart import UnetCooAtt
from model.ResUnet import ResUnet
from model.UnetSPAtt import UnetSpAtt

def train_net(net, device, loss_name, start_epoch, model_name, weight_path, data_path, criterion, loss_history,
              epochs=40,
              batch_size=2,
              lr=0.00001):
    # 加载训练集,此时的训练图片维度为1*H*W
    isbi_dataset = ISBI_Loader(data_path, isTrain=True, isVal=False)
    val_dataset = ISBI_Loader(data_path, isTrain=False, isVal=True)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.99))
    # 训练epochs次
    train_iteration = len(train_loader)
    val_iteration = len(val_loader)
    if arg.load_pretrain_model:
        print('Load weights {}.'.format(weight_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = net.state_dict()
        pretrained_dict = torch.load(weight_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(net, device,loss_name, model_name, criterion, optimizer, epoch, epochs, train_loader,
                                     train_iteration)
        val_loss = val_one_epoch(net, device,criterion, optimizer, epoch, epochs, val_loader, val_iteration)
        print(f'Total Loss: {train_loss:.3f} || Val Loss: {val_loss:.3f} ')
        loss_history.append_loss(train_loss, val_loss)


def train_one_epoch(net, device,loss_name, model_name, criterion, optimizer, curEpoch, endEpoch, train_loader, iteration):
    print('start train:')
    with tqdm(total=iteration, desc=f'Epoch {curEpoch + 1}/{endEpoch}', postfix=dict, mininterval=0.3) as pbar:
        total_loss = 0
        net.train()

        for image, label in train_loader:
            optimizer.zero_grad()

            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            dice_coe = dice_coeff(pred, label)
            loss = criterion(pred, label)
            total_loss = total_loss + loss
            label = label.flatten().detach()
            pred = pred.flatten().detach()

            # label = label.cpu().detach().numpy()  # type:np.ndarray
            # pred = pred.cpu().detach().numpy()  # type:np.ndarray
            # label = label.astype(int)
            # pred = pred.astype(int)
            _f_score = f1_lossFromTensor(pred, label)
            print(
                f'current_epoch:{curEpoch + 1}/Epochs{endEpoch}=========>loss:{loss.item():.3f},f_score:{_f_score:.3f},dice_coe:{dice_coe:.3f}')
            save_dir = f'logs/{model_name}_{loss_name}'
            check_dir_and_create(save_dir)
            torch.save(net.state_dict(), os.path.join(save_dir, f'{model_name}_{curEpoch}_{endEpoch}.pth'))
            # 更新参数
            loss.backward()
            optimizer.step()
    pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration),
                        'lr': get_lr(optimizer)})
    pbar.update(1)
    return total_loss.item() / (iteration)


def val_one_epoch(net, device,criterion, optimizer, curEpoch, endEpoch, train_loader, iteration):
    print('start val:')
    with tqdm(total=iteration, desc=f'Epoch {curEpoch + 1}/{endEpoch}', postfix=dict, mininterval=0.3) as pbar:
        total_loss = 0
        net.eval()

        for image, label in train_loader:
            optimizer.zero_grad()

            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            dice_coe = dice_coeff(pred, label)
            loss = criterion(pred, label)
            total_loss = total_loss + loss
            label = label.flatten().detach()
            pred = pred.flatten().detach()

            _f_score = f1_lossFromTensor(pred, label)
            print(
                f'current_epoch:{curEpoch + 1}/Epochs{endEpoch}=========>loss:{loss.item():.3f},f_score:{_f_score:.3f},dice_coe:{dice_coe:.3f}')
            # 保存loss值最小的网络参数

            # 更新参数
            loss.backward()
            optimizer.step()
            # loss_history.append_loss(total_loss / (start_epoch + 1), val_loss / (epoch_step_val + 1))
        # print(f'interations:{iteration}')
    pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration),
                        'lr': get_lr(optimizer)})
    pbar.update(1)
    return total_loss.item() / (iteration)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument('--model', '-a', metavar='MODEL', default='UnetSPAtt', type=str,
                       help='Choose the model')
    parse.add_argument('--weight_path', type=str, default='logs/Unet_CBAM_37.pth', help='load pre_train weight_path')
    parse.add_argument('--load_pretrain_model', type=bool, default=False, help='decide the model whether load weight')
    parse.add_argument("--batch_size", type=int, default=6)
    parse.add_argument('--data_path', default='COVID', help='the data root')
    parse.add_argument("--log_dir", default='logs', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    parse.add_argument('--start_epoch', type=int, default=0, help='train start epoch')
    parse.add_argument('--end_epoch', type=int, default=70, help='the train epochs = endEpoch - startEpoch')
    parse.add_argument('--lr', type=float, default=0.00001, help='the learning rate')
    parse.add_argument('--loss_function', type=str, default='BCE', help='choose the loss function')
    args = parse.parse_args()
    return args


def check_dir_and_create(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        return False
    return True


def get_lr_parameter(arg):
    _, loss, _ = get_log_model_path(arg)
    criterion = nn.BCELoss().cuda()
    if loss == 'BCE':
        criterion = nn.BCELoss().cuda()
    elif loss == 'BCEDice':
        criterion = BCEDiceLoss().cuda()
    elif loss == 'Dice':
        criterion = SoftDiceLoss().cuda()
    return criterion


def get_log_model_path(arg):
    loss = arg.loss_function
    log_dir = arg.log_dir
    model_name = arg.model
    log_model_name = model_name + '_' + loss
    log_model_dir = os.path.join(log_dir, log_model_name)
    check_dir_and_create(log_model_dir)
    return log_model_dir, loss, model_name


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_model(arg):
    model_name = arg.model
    net = UNet(n_channels=1, n_classes=1)
    if model_name == 'UnetCBAM':
        net = UnetCBAM(n_channels=1, n_classes=1)
    elif model_name == 'UnetCooAtt':
        net = UnetCooAtt(in_channle=1, n_classes=1)
    elif model_name == 'Unet':
        net = UNet(n_channels=1, n_classes=1)
    elif model_name == 'ResUnet':
        net = ResUnet(in_channel=1, n_classes=1)
    elif model_name == 'UnetSPAtt':
        net = UnetSpAtt(in_channle=1,n_classes=1)
    device = get_device()
    net.to(device=device)
    return net

def get_weight_path(arg):
    weight_path = arg.weight_path
    load_weight = arg.load_pretrain_model
    if not load_weight:
        weight_path = ''
    return weight_path

if __name__ == "__main__":
    arg = getArgs()
    # 指定训练集地址，开始训练
    batch_size = arg.batch_size
    data_path = arg.data_path
    start_epoch = arg.start_epoch
    criterion = get_lr_parameter(arg)
    net = get_model(arg)
    weight_path = get_weight_path(arg)
    end_epochs = arg.end_epoch
    log_dir, loss, model_name = get_log_model_path(arg)
    loss_history = LossHistory(log_dir)
    train_net(net, get_device(), loss, start_epoch, model_name, weight_path, data_path, criterion, loss_history,
              epochs=end_epochs,
              batch_size=batch_size)
