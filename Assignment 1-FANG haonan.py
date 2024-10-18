import cv2
import pytesseract
import numpy as np
import pytesseract
from pytesseract import Output
import re

img = cv2.imread('image.jpg')
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img, config=custom_config)
#print(text)


def get_grayscale(image): #将输入的彩色图像image转换为灰度图像
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image): #使用中值滤波去除噪声
    return cv2.medianBlur(image,5)

def thresholding(image):#阈值处理
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def opening(image):#开运算
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):#边缘检测
    return cv2.Canny(image, 100, 200)

def match_template(image, template):#模板匹配，返回一个表示匹配程度的结果矩阵
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

image = cv2.imread('image.jpg')#读取了一个名为'image.jpg'的图像文件，将其存储在变量image中
gray = get_grayscale(image)#调用get_grayscale函数将彩色图像image转换为灰度图像，并将结果存储在变量gray中
remove= remove_noise(gray)#调用remove_noise函数对灰度图像gray进行噪声去除处理
thresh = thresholding(gray)#调用thresholding函数对灰度图像gray进行阈值处理
opening = opening(gray)#调用opening函数对灰度图像gray进行开运算处理
canny = canny(gray)#调用canny函数对灰度图像gray进行边缘检测处理

'''
#灰度处理
img = gray  #将灰度处理的得到的图像数据存储在img中
cv2.namedWindow('gray-image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('gray-image',400, 600)
cv2.imshow('gray-image',img) #使用 OpenCV 的imshow函数显示图像
cv2.waitKey(0) #这个函数会使程序暂停，等待用户的键盘输入。参数0表示无限等待，直到有键盘按键被按下。
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)



#进行噪声去除
img = cv2.imread('img_2.png')
img = remove_noise(img)
cv2.namedWindow('denoised-image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('denoised-image',300, 300)
cv2.imshow('denoised-image', img)
cv2.waitKey(0)



#阈值处理
img = thresh
cv2.namedWindow('thresh-Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('thresh-Image',400, 600)
cv2.imshow('thresh-Image', img)
cv2.waitKey(0)
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)



#开运算处理
img = opening
cv2.namedWindow('opening-image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('opening-image',400, 600)
cv2.imshow('opening-image',img)
cv2.waitKey(0)
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)



#边缘检测处理
img = canny
cv2.namedWindow('canny-image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('canny-image',400, 600)
cv2.imshow('canny-image',img)
cv2.waitKey(0)
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)
'''


d = pytesseract.image_to_data(img, output_type=Output.DICT) #对图像进行光学字符识别，并以字典的形式返回识别结果
#print(d.keys()) #打印出这个字典的键



# 在图像中识别文本区域并为其绘制矩形框
h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img) #使用pytesseract库的image_to_boxes函数对图像img进行处理。这个函数会识别图像中的文本，并返回每个字符的边界框信息
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',400, 600)
cv2.imshow('image',img)
cv2.waitKey(0)


#在文本单词周围设置框
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',400, 600)
cv2.imshow('image',img)
cv2.waitKey(0)



#从图像中识别特定格式的日期文本并为其绘制矩形框
img = cv2.imread('image2.jpg')
d = pytesseract.image_to_data(img, output_type=Output.DICT)
keys = list(d.keys())
# 正则表达式模式,“YYYY-MM-DD” 的字符串，其中年份以 19 或 20 开头，月份在 01 到 12 之间，日期在 01 到 31 之间
date_pattern = '^(19|20)\d\d-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])$'
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
    	if re.match(date_pattern, d['text'][i]):
	        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',400, 600)
cv2.imshow('image',img)
cv2.waitKey(0)



#从图像中提取数字并输出
img2 = cv2.imread('image2.jpg')
custom_config = r'--oem 3 --psm 6 outputbase digits'
print(pytesseract.image_to_string(img2, config=custom_config))
cv2.waitKey(0)


