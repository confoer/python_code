import random#导入“随机”库
import time#导入“时间”库

random.seed(1)

piplinein = []#输入管道
piplineout = []#输出管道

for i in range(10):#重复10次
    piplinein.append(random.randint(0, 30))#在0到30之间随机生成一个整数，并添加到输入管道中

# print(piplinein)
# print(piplineout)
################################
myhand = 0#寄存器，我把它叫做手
stock0 = 0#寄存器，也许会有用，也许不会
stock1 = 0
stock2 = 0
stock3 = 0
####################给定函数############

def takein():
    global myhand
    global piplinein
    
    myhand = piplinein.pop()
    
    piplineout.append(myhand)
    
def giveout():
    global myhand
    global piplineout
    piplineout.insert(0, myhand)
    
def addone():
    global myhand
    myhand += 1
    
def subone():
    global myhand
    myhand -= 1
    
def add(n):
    global myhand
    global stock0
    global stock1
    global stock2
    global stock3
    
    if n == 0:
        myhand += stock0
    elif n == 1:
        myhand += stock1
    elif n == 2:
        myhand += stock2
    elif n == 3:
        myhand += stock3
    else:
        print("stock error")
        
def savetostock(n):
    global myhand
    global stock0
    global stock1
    global stock2
    global stock3
    if n == 0:
        stock0 = myhand
    elif n == 1:
        stock1 = myhand
    elif n == 2:
        stock2 = myhand
    elif n == 3:
        stock3 = myhand
    else:
        print("stock error")
        
def loadfromstock(n):
    global myhand
    global stock0
    global stock1
    global stock2
    global stock3
    if n == 0:
        myhand = stock0
    elif n == 1:
        myhand = stock1
    elif n == 2:
        myhand = stock2
    elif n == 3:
        myhand = stock3
    else:
        print("stock error")

#############################3###
while True:#大家把这个循环叫做主循环，我喜欢把它叫做“时间”，空间是由重复产生的，如果你把时间想象成空间，那时间也是由重复产生的
    time.sleep(1)#休息一秒
    if len(piplinein) == 0:#防止溢出
        print("piplinein is empty")
        break#跳出循环/跳出时间
    ##########################！！编辑区域！！############################
    
    #takein()
    # giveout()
    # savetostock(0)
    # add(0)
    #????????
    #booth算法，需要3个寄存器，我的手也算一个
    #计算机组成原理必考
    
    
    #如果流水线不可能工作请联系生产商
    
    ####################################################################
    print("########################")
    print(piplinein)
    print("-----------------------")
    print(f"{stock0}||{stock1}||{stock2}||{stock3}")
    print("-----------------------")
    print(piplineout)
    print("########################")
    