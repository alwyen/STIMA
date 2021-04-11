import numpy as np
from PIL import Image
import os

def world_coord(baseline, f, X_l, X_r):
    X_list = list()
    Z_list = list()

    assert len(X_l) == len(X_r)

    for i in range(1, len(X_l)):
        X = baseline*i*X_l[i]/(X_l[i] - X_r[i])
        Z = baseline*i*f/(X_l[i] - X_r[i])
        X_list.append(X)
        Z_list.append(Z)
    
    X = np.array(X_list)
    Z = np.array(Z_list)

    return X, Z

if __name__ == '__main__':
    baseline = 1.625
    f = 3074

    near = np.array([3384, 3032, 2640, 2280, 1896, 1480, 1100, 774, 384])
    far = np.array([2976, 2816, 2628, 2464, 2276, 2060, 1896, 1732, 1568])

    X_l_near = np.full(len(near), near[0])
    X_r_near = np.copy(near)

    X_l_far = np.full(len(far), far[0])
    X_r_far = np.copy(far)

    X_near, Z_near = world_coord(baseline, f, X_l_near, X_r_near)
    X_far, Z_far = world_coord(baseline, f, X_l_far, X_r_far)

    print(Z_near)
    print(Z_far)