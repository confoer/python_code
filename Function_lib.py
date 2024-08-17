import pyautogui
import pyperclip
import shutil
from cnocr import CnOcr
import time
import json
import random
import requests
#if __name__ =="__main__":#程序开始运行处

def get_coordinates():#获取鼠标坐标
    coordinates = [0,0,0]
    i = 0
    while True:
        coordinates[i]=pyautogui.position() #每过一秒把坐标存到数组中
        i = i+1
        if i ==3:
            i=0
        if coordinates.count(coordinates[i])==3 and coordinates[i]!=0:
            return coordinates[i]
        time.sleep(1)

def capture_and_ocr(start,end):#获取图像和OCR识别
    ocr = CnOcr()
    screenshot_filename = "mix.jpg"
    r=[min(start[0],end[0]),min(start[1],end[1]),abs(start[0]-end[0]),abs(start[1]-end[1])]
    pyautogui.screenshot(imageFilename=screenshot_filename,region=r)
    out=ocr.ocr("mix.jpg")
    for i in out:
        print(i["text"])

def optimize_ocr():#OCR识别(识别网页信息)
    regions =[
        ()
    ]
    ocr = CnOcr(model_name='conv-lite-fc')#conv-lite-fc:综合中英文识别模型#实例化
    all_text = []#存储所有区域的文字识别结果
    for i,region in enumerate(regions):#enumerate同时获得变量的索引和值
        image = "paper{}.jpg".format(i)
        if screenshot_and_save(region=region,filepath=image):
            out = ocr.ocr(image)
            for item in out:
                if item["score"]>0.4:
                    all_text.append(item["text"])
    with open("deta.json","w","utf-8") as f :
        #使用json库的dump方法将all_text列表写入f中
        json.dump(all_text,f,ensure_ascii=False)

def screenshot_and_save(region,filepath):#截图并保存
    try:    
        screenshot = pyautogui.screenshot(region=region)
        screenshot.save(filepath)
        return True
    except:
        pyautogui.alert(text="截图保存失败",title="警告",button="收到")
        return False

def location_and_click(img_path,clicks=1,confidence=0.9):#确认坐标并打开软件
    pg=pyautogui
    location = pg.locateCenterOnScreen(img_path,confidence=confidence)
    pg.moveTo(location.x,location.y,duration=1)

    pg.click(location.x,location.y,clicks=clicks)

def copy_file(src_path, dst_path): #复制文件到指定路径 
    """参数:  src_path (str): 源文件路径。 dst_path (str): 目标文件路径（包括文件名）。  """  
    try:  
        shutil.copy(src_path, dst_path)  
        print(f"文件 {src_path} 已成功复制到 {dst_path}")  
    except IOError as e:  
        print(f"在复制文件时发生错误: {e}")   