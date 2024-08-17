import cv2
import numpy as np


image = cv2.imread("D:\\1.jpg")# 输入图像路径
if image is None:
    print("Error: Unable to load image.")
    exit()


hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 红色HSV范围
low_range = np.array([0, 86, 0])  
high_range = np.array([250, 255, 255])


th = cv2.inRange(hue_image, low_range, high_range)


img = np.zeros(image.shape, np.uint8)
img[:, :] = (255, 255, 255)


img[th == 255] = image[th == 255]


cv2.imshow('original_img', image)
cv2.imshow('extract_img', img)
cv2.imwrite("out.jpg",img)


cv2.waitKey(0)
cv2.destroyAllWindows()
