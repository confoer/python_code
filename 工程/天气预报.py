import requests
import pyautogui

x = pyautogui.prompt(text="请输入您要查询的城市:",title="天气预报",default="按Cancel取消")
if x == None:
    pyautogui.alert(text="未输入城市！！",title="警告",button="好的")
else:
    a = "http://v1.yiketianqi.com/api?appid=68889179&appsecret=PKbLPW3e&city="+x
    b = requests.get(a)
    c = b.json()
    city = c["city"]
    data = c["data"]
    data1 = data[0]
    index = data1["index"]
    index1 = index[1]
    index2 = index[2]
    index3 = index[3]
    desc1 = index1["desc"]
    desc2 = index2["desc"]
    desc3 = index3["desc"]
    day = data1["day"]
    wea = data1["wea"]
    air_level = data1["air_level"]
    air_tips = data1["air_tips"]
    pyautogui.alert(text=f"{city}\n{day}\n{wea}\n{air_level}\n{air_tips}\n{desc1}\n{desc2}\n{desc3}",title="天气状况")

   