import img2pdf
 
images = ["111.jpg"]#输入路径
output = "111.pdf"#输出的名字
 
with open(output, "wb") as f:
 write_content = img2pdf.convert(images)
 f.write(write_content) # 写入文件
print("转换成功！") # 提示语