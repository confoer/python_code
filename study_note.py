import time      #导入时间库
import json      #导入json库，将json格式转化为python格式
import random    #随机生成
import requests  #网络请求的库
from bs4 import BeautifulSoup #从网页抓取数据的库
import matplotlib.pyplot as plt

#故障保护
#1，鼠标移动到四个角
#2，ctrl+shift+alt

import pyautogui #自动化库

#得到屏幕尺寸
pyautogui.size()

#获取当前坐标
pyautogui.position()

print(pyautogui.onScreen(1920,331))#判断坐标是否正确
while True:#鼠标坐标实时跟踪器
    y = pyautogui.position()#获取当前坐标
    print(y)
    time.sleep(0.3)
    
pyautogui.moveTo(x=1100,y=1230)#鼠标瞬移到指定坐标
pyautogui.moveTo(x=1100,y=1230,duration=8)##鼠标以固定时间移动到指定坐标

pyautogui.moveRel(xOffset=-430,yOffset=-45,duration=3)#相对移动

pyautogui.drag(xOffset=-309,yOffset=20,duration=3)#相对拖拽
pyautogui.dragTo(x=100,y=98,duration=5)#绝对拖拽

for i in range(50):
    pyautogui.moveTo(220,450)
    time.sleep(3)
    pyautogui.moveTo(300,770)
    time.sleep(3)

pyautogui.click(x=27,y=800)#鼠标点击指定坐标

for i in range(100):#鼠标连点器
    pyautogui.click(x=1300,y=1340)
    time.sleep(0.6)

pyautogui.scroll(50)#鼠标滚动

pyautogui.write("hello")#在光标闪烁的地方输入英文

import pyperclip #复制粘贴库

time.sleep(7)
for i in range(34):
    pyperclip.copy("你好")#将内容复制到粘贴板上
    pyautogui.hotkey("ctrl","v")#粘贴到光标闪烁的地方
    pyautogui.hotkey("enter")#按下回车
pyautogui.alert(text="1",title="2",button="3")#警告框
pyautogui.confirm(text="Yes or No",title="chioce",buttons=["Yes","No","其他"])#选择框
pyautogui.prompt(text="请输入数字",title="你多大了",default="")#明文输入框
pyautogui.password(text="密语",title="树洞",default="",mask="*")#密文输入框

from cnocr import CnOcr #导入CnOcr(图像识别库)(光学字符识别)

pyautogui.screenshot("file.png")#全屏截图
##图像识别
x = pyautogui.locateCenterOnScreen("3.png",confidence=0.75)#confidence:匹配度#找到满足的第一个图标
pyautogui.click(x,clicks=2)#双击坐标
#cnocr四部曲：1，导入库 2，实例化 3，传图片
CnOcr(det_model_name="nalve_det")#det_model_name:检测模型，适合文字特别整齐时使用
CnOcr(rec_model_name="ch_pp-OCRv3")#rec_model_name:处理竖排文字

#单行文字识别
x = CnOcr()
y = x.ocr_for_single_line("图片路径")

pyperclip.paste()#从剪贴板中获取内容
pyperclip.waitForPaste()#查看剪贴板是否有东西，没有则等待
pyperclip.waitForNewPaste()#等粘贴板中有东西再打印

#abs:求绝对值
#screenshot(imageFilename="x.jpg")保存截图为x
#传入region函数截取区域

#duration：移动时间
#url:网址

a = requests.get("https://www.baidu.com/")
print(a.status_code)#输出请求后，返回的状态码
print(a.headers)#响应头部内容
print(a.text)#输出文本内容


h = {#身份信息(在beautiful soup里使用)
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': "1",
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-User': '?1',
    'Sec-Fetch-Dest': 'document',
    }
pandas.DataFrame()#将数据格式转化为dataframe格式
#变量名称.to excel("文件名字.xlsx",index="")#如果自带表头，则填False
requests.get(url="",params="",headers="")#params:查询条件.headers:头部信息（用于攻破反爬技术）
#data_excel = pandas.read_excel("表格路径",None):读取表格数据并存储在变量中（默认读取sheet1数据）。none:表示读取全表数据

#dataframe：二维表格数据类型
# serises：一维数据类型，只有表格的行或列