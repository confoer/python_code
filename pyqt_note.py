import sys
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QLabel,QLineEdit,QDesktopWidget,QVBoxLayout,QGroupBox,QMainWindow
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class test():
    app = QApplication(sys.argv)
    
    w = QWidget()
    
    w.setWindowTitle("第一个PyQt")#设置窗口标题
    w.resize(1200,600)

    # 创建按钮
    # btn = QPushButton("按钮")
    # btn.setParent(w)# 添加到窗口显示

    # 创建纯文本
    label = QLabel("账号:",w)# 创建一个(纯文本)，创建时指定父对象
    label.setGeometry(200,200,300,200)# 显示位置与大小:x,y,w,h

    # 创建文本框
    edit =QLineEdit(w)
    edit.setPlaceholderText("请输入账号")
    edit.setGeometry(550,200,200,20)

    # 窗口设置在屏幕左上角
    w.move(0,0)
    
    # 调整窗口在屏幕中央显示
    center_pointer = QDesktopWidget().availableGeometry().center()# 获取屏幕中央坐标
    x = center_pointer.x()
    y = center_pointer.y()
    # w.move(x-600,y-300)
    # print(w.frameGeometry())
    print(w.frameGeometry().getRect())
    print(type(w.frameGeometry().getRect()))
    old_x,old_y,width,height = w.frameGeometry().getRect()
    w.move(x -  width//2,y - height//2)

    # 设置图标
    # w.setWindowIcon(QIcon("icon.png"))

    w.show()# 展示窗口 
    app.exec()#程序进行循环等待状态

class MyWindowV(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(300,300)
        self.setWindowTitle("垂直布局")

        layout = QVBoxLayout()
        btn1 = QPushButton("按钮1")
        btn2 = QPushButton("按钮2")
        btn3 = QPushButton("按钮3")

if __name__ =='__main__':
    app = QApplication(sys.argv)
    w = MyWindowV()
    w.show()
    app.exec()