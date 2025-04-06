from cnocr import CnOcr

img  = f"img\\111.png"
ocr =CnOcr()
result = ocr.ocr(img)
for i in range(len(result)):
    print(result[i]['text'])