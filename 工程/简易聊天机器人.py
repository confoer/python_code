import requests   #导库

while True:
    url = "	http://api.qingyunke.com/api.php?key=free&appid=0&msg="  #寻址

    word = input("你说：")

    res = requests.get(url+word)  #请求,返回res可以想象成网购到的包裹

    # print(res.text)#适用于文本、json格式内容
    # print(res.content)#图片、音乐、文件、电影……
    res2 = res.json()  #直接转化json格式内容为python字典
    print("回答：",res2["content"])