a = [{"width": 0},
     {"width": 56},
     {"width": 3},
     {"width": 9}
     ]


def so(a):
    return a["width"]


b = sorted(a, key=(lambda data: data["width"]))
print(b)
print(a)
