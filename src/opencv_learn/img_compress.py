# -*- coding:utf-8 -*-
import cv2

# img = cv2.imread('F:\\work\\catch\\2.jpg')
# cv2.imwrite('F:\\work\\catch\\test_compress.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 30))


img = cv2.imread('/media/chenhao/package/dataset/cat_dogs/PetImages/Dog/0.jpg')

cv2.namedWindow("Image")
cv2.imshow('Image', img)
cv2.waitKey(0)