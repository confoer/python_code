# list(map(int,input().split()))# split()方法用于将字符串分割成多个部分，默认以空格作为分隔符。map()函数会将每个字符串转换为整数。

# python除法计算默认float,整除运算根据题目给的信息是向下整除，但是使用//整除是过不了所有的测试点的。最后是使用int()强制类型转换才通过的。

# f"{c:.3f}")#:指定格式
# %g:根据余数的大小自动选择最合适的表示法。
# number.sort()# 对列表进行排序
# "%.9f"%a:保存a的9位小数
# print(i,end=' ') # 结束添加空格
# x = input()
# y = len(x)
# z = str(y)
# # 字符串倒序输出
# i = y-1
# i = int(i)
# while i >= 0:
#     print(x[i],end='')
#     i-=1
# 倒序输出
# print(n[::-1])

# 从1970年1月1日0时0分0秒开始的第n毫秒是几时几分几秒？
from datetime import datetime, timedelta

start = datetime(year=1970, month=1, day=1)# 1970年1月1日0时0分0秒
dela = timedelta(milliseconds=1)# 1毫秒间隔
now = int(input())

now = start + now * dela# 计算时间
print('%02d:%02d:%02d' % (now.hour, now.minute, now.second))
for i in da:
    da[i] = da.get(i, 0) + 1 # 实现了对字典的键值操作，如果 i 不存在于字典中，则默认值为 0，然后加 1。