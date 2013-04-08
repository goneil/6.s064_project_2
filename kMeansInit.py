import numpy as np

def kMeansInit(X, k, p0):
    def distance(x1, x2):
        return np.linalg.norm(x1 - x2)
    def comp(x1, x2):
        dist1 = distance(x1, p0)
        dist2 = distance(x2, p0)
        if dist1 > dist2:
            return 1
        elif dist1 < dist2:
            return -1
        else:
            return 0
    lst = sorted(X, comp)
    lst.reverse()
    ret_list = [p0]
    for i in range(k - 1):
        ret_list.append(lst[i])
    return np.array(ret_list)
        


