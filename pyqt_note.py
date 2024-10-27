import sys

from PyQt5.QtWidgets import QApplication,QWidget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    w = QWidget()
    
    w.setWindowTitle("第一个PyQt")#设置窗口标题
    
    w.show()# 展示窗口
    
    app.exec_()#程序进行循环等待状态