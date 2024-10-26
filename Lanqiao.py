# list(map(int,input().split()))# split()方法用于将字符串分割成多个部分，默认以空格作为分隔符。map()函数会将每个字符串转换为整数。

# python除法计算默认float,整除运算根据题目给的信息是向下整除，但是使用//整除是过不了所有的测试点的。最后是使用int()强制类型转换才通过的。

# ord()# 字符串转整数
# chr()# 整数转字符串
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

t = int(input())
arr = []
for i in range(t):
    arr+=[list(map(int,input().split()))]