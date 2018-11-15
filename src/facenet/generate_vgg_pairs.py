# -*- coding:utf-8 -*-
"""
this file is use to generate vgg test pairs,see lfw pairs.txt
"""
import os
import numpy as np


def generate_pairs():
    poch = 10
    groupnum = 300
    batch_size = 30

    path = "F:\\dataset\\vggface2_test\\test"
    class_list = os.listdir(path)
    print(len(class_list))
    print(class_list)
    # 随机选择300个，制作pair
    random_class = np.random.choice(class_list, groupnum, False)
    # 不随机
    # random_class = class_list

    # 分成10个批，
    with open("vgg_pairs,txt", "w") as vp:
        for i in range(poch):
            start = i * batch_size
            end = i * batch_size + batch_size
            print('start end', start, end)
            # 300 个分成10批，一个批30人。每个批要生成 300个同一个人，300个不同人的 标记 共6000
            batch_class_list = random_class[start:end]
            # print('batch_class_list:')
            # print(batch_class_list)
            # 对于每个人 生成10条相同人脸的标签，一批30个人生成300条
            negative_label = []

            for n_idx in range(len(batch_class_list)):
                name = batch_class_list[n_idx]
                person_path = os.path.join(path, name)
                # print(person_path)
                person_imgs = os.listdir(person_path)
                # print('person_imgs ', person_imgs)
                # C(5,2) 选择10个
                five_pic = np.random.choice(person_imgs, 5, False)
                for m in range(len(five_pic)):
                    n = m + 1
                    while n < len(five_pic):
                        positive_str = name + ' ' + five_pic[m] + ' ' + five_pic[n] + '\n'
                        n = n + 1
                        # print(positive_str)
                        vp.write(str(positive_str))
                # 对30个人生成条不同
                m_idx = n_idx + 1
                while m_idx < len(batch_class_list):
                    person2_name = batch_class_list[m_idx]
                    person2_path = os.path.join(path, person2_name)
                    person2_imgs = os.listdir(person2_path)
                    negative_str = name + ' ' + person_imgs[0] + ' ' + person2_name + ' ' + person2_imgs[0] + '\n'
                    if len(negative_label) < 300:
                        negative_label.append(negative_str)
                        vp.write(str(negative_str))
                    # print(negative_label)
                    m_idx = m_idx + 1


generate_pairs()
