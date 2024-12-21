a = int(input())
b = [1,1]
n=2
sum = 0
while(True):
    sum = b[n-1]+b[n-2]
    if sum >a :
        break
    b.append(sum)
    n+=1
for i in b:
    print(i)