from pdf2docx import Converter
from docx2pdf import convert
import win32com.client as win32
import os
import pyautogui
# PDF转WORD
def pdf_to_word(file_path):
    pdf =file_path #输入要转换的文件
    f=open(pdf,'r')
    doc_name_choice = input('你确定需要转换吗，确定就属于y,不转换输入n:')
    if doc_name_choice =='y' or doc_name_choice =='Y':
            doc_name = input('输入你转换后存储的word文件名称:')+'.docx'
    else:
            pdf_name = os.path.basename(pdf)
            doc_name = os.path.splitext(pdf_name)[0]+'.docx'
    cv = Converter(pdf)
    path = os.path.dirname(pdf)
    cv.convert(os.path.join(path, "", doc_name),start=0, end=None)
    print('WORD')
    cv.close()

# WORD转PDF
def word_to_pdf(file_path):
    """
    将一个 .doc 或 .docx 文件转换为 .pdf 文件。
    
    :param doc_path: 输入的 .doc/.docx 文件路径
    :param pdf_path: 输出的 .pdf 文件路径
    """
    # 确保文件路径存在
    if not os.path.exists(file_path):
        print(f"错误：无法找到文件 {file_path}")
        return False

    # 创建 Word 应用程序对象
    word = win32.gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Open(file_path)
    
    try:
        # 将文档另存为 PDF
        doc.SaveAs(file_path, FileFormat=17)  # FileFormat=17 表示 PDF 格式
        print(f"成功将 {file_path} 转换为 {file_path}")
        return True
    except Exception as e:
        print(f"转换失败：{str(e)}")
        return False
    finally:
        # 关闭 Word 文档和应用程序
        doc.Close()
        word.Quit()
while(True):
	if __name__ =='__main__':
		print("请输入你想选择的模式")
		print("1.pdf转word 2.word转pdf")
		x=input()
		a=int(x)
		if a==1:	
			b=pyautogui.prompt(text="路径为",title="请输入文件路径",default="")
			pdf_to_word(b)
			b=input("是否继续转换文件格式(y/n):")
			if b=='y' or b=='Y':
				continue
			else:
				break
		elif a==2:
			c=pyautogui.prompt(text="路径为",title="请输入文件路径",default="")
			word_to_pdf(c)
			c=input("是否继续转换文件格式(y/n):")
			if c =='y' or c =='Y':
				continue
			else:
				break
		else:
			print("输入错误,请重新输入")
			continue