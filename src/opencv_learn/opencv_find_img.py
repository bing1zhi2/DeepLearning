# -*- coding:utf-8 -*-
import cv2
import os

base_dir = '/media/chenhao/study/tmp/testimgs'
base_dir = 'F:/tmp/testimgs'
src_img = os.path.join(base_dir, '023032131_K9783_31_2_30.jpg')
src_img2 = os.path.join(base_dir, '004654573_K505196_0256_2_29.jpg')
# print(src_img)
img = cv2.imread(src_img)
img2 = cv2.imread(src_img2)
# print(img)
template = cv2.imread(os.path.join(base_dir, '333.jpg'))
# print(template)
h, w = template.shape[:2]  # rows->h, cols->w

print(h, w)


# 相关系数匹配方法：cv2.TM_CCOEFF
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

res2 = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)

print(min_val, max_val, min_loc, max_loc)
print(min_val2, max_val2, min_loc2, max_loc2 )

left_top = max_loc  # 左上角
right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置

# cv2.namedWindow("Image")
# cv2.imshow('Image', img)
# cv2.waitKey(0)

cv2.imwrite('aaa.jpg', img)

