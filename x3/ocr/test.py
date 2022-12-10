import cv2 as cv
import numpy as np
 
src = cv.imread("/data/dataset/sample_img/1665383932606.jpg")
cv.imshow("input", src)
h, w, c = src.shape
 
# 手工绘制ROI区域
mask = np.zeros((h, w), dtype=np.uint8)
x_data = np.array([124, 169, 208, 285, 307, 260, 175])
y_data = np.array([205, 124, 135, 173, 216, 311, 309])
pts = np.vstack((x_data, y_data)).astype(np.int32).T
print(pts.shape)
cv.fillPoly(mask, [pts], (255), 8, 0)
cv.imshow("mask", mask)
 
# 根据mask，提取ROI区域
result = cv.bitwise_and(src, src, mask=mask)
cv.imshow("result", result)
cv.waitKey(0)