import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from PIL import Image
class ISBI_Loader(Dataset):
    def __init__(self, data_path, isTrain:bool=True, isVal:bool=False):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path

        if isTrain:
            self.imgs_path = glob.glob(os.path.join(data_path, 'train/*.png'))
            self.flag = 'train'
        elif isVal:
            self.imgs_path = glob.glob(os.path.join(data_path,'val/*png'))
            self.flag = 'val'
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        if self.flag == 'train':
            label_path = image_path.replace('train', 'trainLabel')
        elif self.flag == 'val':
            label_path = image_path.replace('val','valLabel')
        #label_path = label_path.replace('jpg', 'png')

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.resize(image, (256, 256))
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader('../COVID')
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=False)

