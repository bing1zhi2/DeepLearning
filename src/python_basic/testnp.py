import numpy as np


a = np.array([[True,False,False,False],
          [True, False, False, False],
          [True, False, False, False]])
aaa =np.where(a > 0.5)
print(a > .5)

b = np.reshape(a > .5, (-1, a.shape[-1])).astype(np.float32)

print(b)
print(np.where(b>0))
print(b[np.where(b>0)])
