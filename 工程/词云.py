import jieba
import wordcloud
from PIL import Image
import numpy as np
from docx import Document

# 读取Word文档
doc = Document('D:\\word.docx')
text = ' '.join([paragraph.text for paragraph in doc.paragraphs])


# 使用 jieba 进行中文分词
words = jieba.cut(text)
text = " ".join(words)

# 打开图片并转换为 numpy 数组用于词云形状
mask = np.array(Image.open('D:\\222.png'))

# 创建词云对象
wc = wordcloud.WordCloud(
    font_path="D:\\HarmonyOS_Sans_SC_Medium.ttf",  # 字体文件路径，根据实际情况修改
    background_color='white',  # 背景颜色
    mask=mask,  # 设置词云形状
    contour_width=1,  # 轮廓宽度
    contour_color='steelblue'  # 轮廓颜色

)

# 生成词云
wc.generate(text)

# 保存词云图片
wc.to_file('wordcloud.png')