import time
t = time.localtime(time.time())
print(t)
print(time.time())

str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(str)
str = time.strftime("%Y%m%d%H%M%S", time.localtime())
print(str)