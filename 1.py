
import cv2 as cv
import numpy as np

src = cv.imread("AI.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpen_image = cv.filter2D(src[:,:,1], cv.CV_32F, sharpen_op)
sharpen_image = cv.convertScaleAbs(sharpen_image)
cv.imshow("sharpen_image", sharpen_image)

h, w = src.shape[:2]
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = src
result[0:h,w:2*w,:] = sharpen_image
cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "sharpen image", (w+10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("sharpen_image", result)
# cv.imwrite("D:/result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
"""
import cv2
import matplotlib.pyplot as plt
src = cv2.imread("AI.jpg")
cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
cv2.imshow("input", src) # plt.imshow(src)
blur_img = cv2.GaussianBlur(src, (0, 0), 5)
usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
cv2.imshow("mask image", usm) #plt.imshow(usm)
import numpy as np
h, w = src.shape[:2]
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = src
result[0:h,w:2*w,:] = usm
cv2.putText(result, "original image", (10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv2.putText(result, "sharpen image", (w+10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
#cv.putText(图像名，标题，（x坐标，y坐标），字体，字的大小，颜色，字的粗细）
cv2.imshow("sharpen_image", result) #plt.imshow(result)
"""




import cv2

img = cv2.imread('AI.jpg')

## way1: can change the shape of the window and close the window 

#cv2.namedWindow('result',0)
#cv2.startWindowThread()
#cv2.imshow('result',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

## way2: can close the window 
##       can not change the shape of the window

cv2.imshow('result.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)