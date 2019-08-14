# -*- coding:utf-8 -*-
import cv2
import os

base_dir = 'F:/tmp/testimgs'

dir1 = 'K9783_31'

path = os.path.join(base_dir, dir1)
km_dir = os.listdir(path)

template = cv2.imread(os.path.join(base_dir, '333.jpg'))




def compare_with_template(src_filename, template):

    img = cv2.imread(src_filename)
    h, w = template.shape[:2]  # rows->h, cols->w

    print(h, w)

    # 相关系数匹配方法：cv2.TM_CCOEFF
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    # res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print(min_val, max_val, min_loc, max_loc)

    left_top = max_loc  # 左上角
    right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
    cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置

    # cv2.namedWindow("Image")
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)


    cv2.imwrite(os.path.basename(src_filename), img)

    return max_val,min_val


values = []
mins = []
names = []
for f in km_dir:
    filename = os.path.join(path, f)
    if os.path.isfile(filename) and f.endswith('.jpg'):
        print(filename)
        max_val, min_val = compare_with_template(filename, template)
        values.append(max_val)
        mins.append(min_val)
        names.append(os.path.basename(filename))



max_min_v = min(values)
max_max_v = max(values)
print('max 最小：', max_min_v, 'max 最大', max_max_v)

min_min_v = min(mins)
min_max_v = max(mins)
print('min 最小：', min_min_v, 'min 最大', min_max_v)

# idx = values.index(min_v)
# print('最大的索引：', idx)
#
# print('对应pic', names[idx + 1])


