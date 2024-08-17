import pyautogui as pg #自动化库
import pyperclip as pc#复制粘贴库
from cnocr import CnOcr #导入CnOcr(图像识别库)(光学字符识别)
import time      #时间
import json      #导入json库
import random    #随机生成

# 关键字列表
keyword_list = ['你好', '哈哈', '我喜欢你']
# 匹配到的关键字回复
reply_list = ['你有事？', '？', '你是个好人']
# 没有匹配到关键字的随机回复
random_speak = ['随便',"哎哟~你干嘛~","尊嘟假嘟","泰酷辣!","退退退！","我真的栓q"]
# 微信截图区域
def region_message():
    location = pg.locateOnScreen("laugh.png",confidence=0.9)
    x,y,w,h=location
    region = (x-20,y-5-433,446,433)
    return region

# 确认坐标并打开微信
def location_and_click(img_path,clicks=1,confidence=0.9):
    location = pg.locateCenterOnScreen(img_path,confidence=confidence)
    pg.moveTo(location.x,location.y,duration=1)

    pg.click(location.x,location.y,clicks=clicks)

# 打开微信后将消息区域的截图交给ocr模型进行识别
def capture_and_ocr(region):
    screenshot = pg.screenshot(region=region)
    screenshot.save("wx_xx.png")
    img_fp = 'wx_xx.png'
    out = ocr.ocr(img_fp)
    text = out[-1]['text']
    return text

# 判断消息是否有关键字
def has_keyword(text,keyword_list):
    for key in keyword_list:
        if key in text:
            return True
    return False

# 发送消息
def send_message(message):
    pc.copy(message)
    pg.hotkey('ctrl','v')
    pg.press('enter')

# 根据是否有关键字进行不同的回复消息
def send_message_based_on_keyword(text, keyword_list, reply_list, random_speak):
    if has_keyword(text, keyword_list):
        for i, key in enumerate(keyword_list):
            if key in text:
                send_message(reply_list[i])
                location_and_click("return.png")
                return
    send_message(random.choice(random_speak))
    location_and_click("return.png")
wx_location = location_and_click('logo.png',clicks=2)
time.sleep(2)
ocr = CnOcr()
# 循环执行任务
while True:
    try:
        xx_location = location_and_click('newMess.png',confidence=0.9,clicks=1)
        time.sleep(2)
        text = capture_and_ocr(region_message())
        print("获得新消息：",text)
        send_message_based_on_keyword(text, keyword_list, reply_list, random_speak)
    except Exception as e:
        print(f"当前没有新的消息，等待10秒后重新执行")

        # 等待5秒再次执行
    time.sleep(5)
