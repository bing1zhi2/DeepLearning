# -*- coding:utf-8 -*-
# bing1   

# 2019/8/29   

# 18:07   
"""
MIT License

Copyright (c) 2019 Hyman Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

preds = np.array([0, 1, 2, 0, 1, 2])
labels = np.array([0, 2, 1, 0, 0, 1])


preds = np.array([4191, 1430, 174, 26, 313, 2556, 1920, 259, 1430, 4120])
labels = np.array([4191, 1430, 174, 26, 313, 2556, 1920, 259, 1430, 4120])

# def calculate_accuracy(predict_issame, actual_issame):
#     tp = np.sum(np.logical_and(predict_issame, actual_issame))
#     fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#     tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
#     fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
#
#     tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
#     fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
#     acc = float(tp + tn) / 9
#     return tpr, fpr, acc
#
# for pred, label in zip(preds, labels):
#     tp = (pred + label == 2).sum()
#     tn = (pred + label == 0).sum()
#     fp = (pred - label == 1).sum()
#     fn = (pred - label == -1).sum()
#
#     print(tp, tn, fp ,fn)


# y_true=[1,2,3]
# y_pred=[1,1,3]
#
# f1 = f1_score( y_true, y_pred, average='micro' )
# p = precision_score(y_true, y_pred, average='micro')
# r = recall_score(y_true, y_pred, average='micro')
#
# print(f1, p , r)


f1 = f1_score(labels, preds, average="micro")
pre = precision_score(labels, preds, average='micro')
recal = recall_score(labels, preds, average='micro')

print(f1, pre, recal)

# tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
cm = confusion_matrix(labels, preds)
print(cm)
print("---------------")

"""
TN 在 [0, 0] ， , TP 在[1,1], FP 在 [1, 0]  FN 在   [0, 1] 
"""

mcm = multilabel_confusion_matrix(labels, preds)
# mcm = multilabel_confusion_matrix(labels, preds)

print("mcm")
print(mcm)

"""
>>> tp = mcm[:, 1, 1]
>>> tn = mcm[:, 0, 0]
>>> fn = mcm[:, 0, 1]
>>> tp, tn
(array([2, 0, 2], dtype=int64), array([3, 5, 2], dtype=int64))
"""
tp = mcm[:, 1, 1]
tn = mcm[:, 0, 0]
fp = mcm[:, 1, 0]
fn = mcm[:, 0, 1]

print("tp", tp)
print("tn", tn)
print("fp", fp)
print("fn", fn)

"""
P=(TP1+TP2+TP3)/(TP1+FP1+TP2+FP2+TP3+FP3)=(1+0+1)/(1+1+0+0+1+0)=0.6666667
R=(TP1+TP2+TP3)/(TP1+FN1+TP2+FN2+TP3+FN3)=(1+0+1)/(1+0+0+1+1+0)=0.6666667
F1 = 2*(0.6666667*0.6666667)/(0.6666667+0.6666667)=0.6666667

"""

precision = 1.0 * np.sum(tp) / max((np.sum(tp) + np.sum(fp)), 10e-20)
recall = 1.0 * np.sum(tp) / max((np.sum(tp) + np.sum(fn)), 10e-20)
# precision = 1.0 * tp / max(tp     + fp, 10e-20)
# recall = 1.0 * tp / max(tp + fn, 10e-20)
f1score = 2. * precision * recall / max(precision + recall, 10e-20)

print(precision, recall, f1score)

tprs=[]
fprs=[]
for x,y,a, b in zip(tp, fn, fp, tn):
    tpr = 1.0* x / max(x+y, 10e-20)
    fpr = 1.0* a / max(a+b, 10e-20)
    print("tpr, fpr", tpr, fpr)
    tprs+=tpr
    fprs += fpr

# tpr = 1.0 * np.sum(tp) / max((np.sum(tp) + np.sum(fn)), 10e-20)
# fpr = 1.0 * np.sum(fp) / max((np.sum(fp) + np.sum(tn)), 10e-20)
# print(tpr, fpr)



x = fprs
y = tprs

x = [0.1,0.2,0.3]
y = [1,2,4]

plt.plot(x,y, lw= 1)
# plt.plot(fpr,tpr)
plt.show()
