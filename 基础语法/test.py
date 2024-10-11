import pygame as pg

pg.init()
ui=pg.display.set_mode((1520,600))
ui.fill((125,0,0))# 设置背景色
img=pg.image.load('img\\111.png')# 设置图层
ui.blit(img,(0,0))# 渲染图层
pg.display.set_caption("测试")# 设置左上角文字
font = pg.font.SysFont('SimHei',22)
text=font.render('设置字体',True,(0,0,255))
ui.blit(text,(100,100))

# while True:
#     for event in pg.event.get():
#         if event.type == pg.QUIT:
#             exit()
#     pg.display.flip()


import random
l = [2,4,5,6,7,8]
x=random.choice(l)# 从列表中随机抽取
print(x)