from torch.utils.data import Dataset
import cv2
import os

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):# 初始化函数，得到图像路径和标签路径
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self,idx):# 根据索引返回图像和标签
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = cv2.imread(img_item_path)
        label  = self.label_dir 
        return img,label
    
    def __len__(self):
        return len(self.img_path)

root_dir = 'datasets\pytorch_test\train'# 图像根目录
ant_label_dir = "ants"
ant_dataset = MyDataset(root_dir,ant_label_dir)