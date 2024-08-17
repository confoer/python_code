import os
import pyautogui as pg
class BatchRename():

    def __init__(self):
        x=pg.prompt(text="路径:",title="请输入图片路径",default="")#明文输入框
        self.path = x  # 图片的路径

    def rename(self):
        filelist = os.listdir(self.path)
        filelist.sort()
        total_num = len(filelist)
        i = 1 
        for item in filelist:
            if item.endswith('.png') or item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.JPG'):
                src = os.path.join(self.path, item)
                dst = os.path.join(os.path.abspath(self.path),str(i)+ '.png')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except Exception as e:
                    print(e)
                    print('rename dir fail\r\n')
            if item.endswith('.mp4'):
                src = os.path.join(self.path, item)
                dst = os.path.join(os.path.abspath(self.path),str(i)+ '.mp4')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except Exception as e:
                    print(e)
                    print('rename dir fail\r\n')
        print('total %d to rename & converted %d jpgs' % (total_num, i))
if __name__ == '__main__':
    demo = BatchRename()  #创建对象
    demo.rename()   #调用对象的方法
