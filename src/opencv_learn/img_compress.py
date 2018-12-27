import cv2

img = cv2.imread('F:\\work\\catch\\2.jpg')
cv2.imwrite('F:\\work\\catch\\test_compress.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 30))