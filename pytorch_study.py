from numpy import imag, piecewise
from sympy import im
from torch.utils.data import Dataset
import cv2
import os

class MyDataset(Dataset):
    def __init__(self,root_dir,image_dir,label_dir):# 初始化函数，得到图像路径和标签路径
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.image_dir)
        self.label = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
        self.label_path = os.listdir(self.label)


    def __getitem__(self,idx):# 根据索引返回图像和标签
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.image_dir,img_name)
        img = cv2.imread(img_item_path)
        label_name  = self.label_path[idx]
        label_item_path = os.path.join(self.root_dir,self.label_dir,label_name)
        label = open(label_item_path,'r',encoding='utf-8').readlines()
        return img,label
    
    def __len__(self):
        return len(self.img_path)

# root_dir = 'datasets\pytorch_test\\train'# 图像根目录
# ant_label_dir = 'ants\\ants_label'
# ant_image_dir = 'ants\\ants_image'
# bee_label_dir = 'bees\\bees_label'
# bee_image_dir = 'bees\\bees_image'
# ant_dataset = MyDataset(root_dir,ant_image_dir,ant_label_dir)
# bee_dataset = MyDataset(root_dir,bee_image_dir,bee_label_dir)
# train_dataset = ant_dataset + bee_dataset

from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('logs')# 创建一个writer对象，指定日志文件夹为logs
# image_path = 'img\makassaridn-road_demo.png'
# img = cv2.imread(image_path)
# writer.add_image('train',img,2,dataformats='HWC')# 将图像写入日志文件夹
# tensorboard --logdir=logs --port=6006 # 启动tensorboard,并指定端口为6006（终端运行）
# for i in range(100):
    # writer.add_scalar("y=2x ",2*i,i)
# writer.close()

from torchvision import transforms
# transform:将图像转换为tensor，并对图像进行归一化处理
# img = cv2.imread('img/makassaridn-road_demo.png')
# to_tensor=  transforms.ToTensor()
# tensor_img = to_tensor(img)

img = cv2.imread('img/makassaridn-road_demo.png')
to_tensor = transforms.ToTensor()
tensor_img = to_tensor(img)
# Normalize:对图像进行归一化处理
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
# Resize:调整图像大小
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(tensor_img)
print(tensor_img[0][0][0])
print(img_norm[0][0][0])
print(img_resize[0][0][0])
