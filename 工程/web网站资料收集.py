import pyautogui as pg 
from cnocr import CnOcr
import time
import json

region =[
    ()# 坐标数组
]
time.sleep(6)
ocr = CnOcr(model_name='conv-lite-fc')#conv-lite-fc:综合中英文识别模型#实例化
all_text = []#存储所有区域的文字识别结果
j = 0
for i in region:
    image = "paper{}.jpg".format(j)
    paper = pg.screenshot(region=i)
    out = ocr.ocr(image)
    paper.save(image)
    for item in out:
        if item["score"]>0.4:
            all_text.append(item["text"])
    j+=1
with open("deta.json","w","utf-8") as f :
    #使用json库的dump方法将all_text列表写入f中
    json.dump(all_text,f,ensure_ascii=False)