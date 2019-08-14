# -*- coding:utf-8 -*-
import cv2
import os
from matplotlib import pyplot as plt

base_dir = '/media/chenhao/study/tmp/testimgs'
base_dir = 'F:/tmp/testimgs'
src_img = os.path.join(base_dir, '023032131_K9783_31_2_30.jpg')
# print(src_img)
img = cv2.imread(src_img)
# print(img)
template = cv2.imread(os.path.join(base_dir, '333.jpg'))
# print(template)
h, w = template.shape[:2]  # rows->h, cols->w

print('template h,w ', h, w)



img2 = img.copy()

print(h,w)
# All the 6 methods for comparison in a list
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for meth in methods:
    print('meth', meth)
    img_temp = img2.copy()
    # Apply template Matching
    res = cv2.matchTemplate(img_temp, template,meth)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print('min Max loc')
    print(min_val, max_val, min_loc, max_loc )

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        print('in min loc')
        top_left = min_loc
    else:
        print('in max loc')
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    print('top_left ', top_left)
    print('bottom_right ', bottom_right)

    cv2.rectangle(img_temp, top_left, bottom_right, 255, 2)

    plt.subplot(121)
    plt.imshow(res,cmap = 'gray')

    plt.title('Matching Result')
    plt.xticks([]),
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(meth)
    plt.show()