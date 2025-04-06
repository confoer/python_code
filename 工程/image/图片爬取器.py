import requests
import pyautogui

url = "https://gd-hbimg.huaban.com/6fba4b7fbb2ea240de2c4dbe504f4b81b42b212d14476-rW3roz_fw658"#输入图片网址的目标源
a = requests.get(url)
with open('1.jpg', 'wb') as file:#保存图片名字为“1.jpg”
    for chunk in a.iter_content(): 
        file.write(chunk)