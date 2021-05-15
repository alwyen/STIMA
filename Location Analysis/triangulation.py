import numpy as np
from PIL import Image
import os

def mean_std(array):
    print(f'Mean: {np.mean(array)}')
    print(f'STD: {np.std(array)}')

#Skip_First = 0 or 1
def world_coord(baseline, f, X_l, X_r, Skip_First):
    X_list = list()
    Z_list = list()

    assert len(X_l) == len(X_r)

    for i in range(Skip_First, len(X_l)):
        X = baseline[i]*X_l[i]/(X_l[i] - X_r[i])
        Z = baseline[i]*f/(X_l[i] - X_r[i])
        X_list.append(X)
        Z_list.append(Z)
    
    X = np.array(X_list)
    Z = np.array(Z_list)

    return X, Z

#compare all images to first image
def reference_to_first(baseline, f, near, far):
    X_l_near = np.full(len(near), near[0])
    X_r_near = np.copy(near)

    X_l_far = np.full(len(far), far[0])
    X_r_far = np.copy(far)

    increments = np.arange(0,len(far),1)
    baselines = np.full(len(far), baseline)*increments


    X_near, Z_near = world_coord(baselines, f, X_l_near, X_r_near, 1)
    X_far, Z_far = world_coord(baselines, f, X_l_far, X_r_far, 1)

    print('Reference to First Image:')
    print(f'Near:')
    mean_std(Z_near)
    print(f'Far:')
    mean_std(Z_far)
    print()

#first & second, second & third, third & fourth, ... , n & n + 1
def consecutive_images(baseline, f, near, far):
    X_l_near = np.copy(near)
    X_r_near = np.copy(near)
    X_l_near = X_l_near[:len(X_l_near)-1]
    X_r_near = X_r_near[1:len(X_r_near)]

    X_l_far = np.copy(far)
    X_r_far = np.copy(far)
    X_l_far = X_l_far[:len(X_l_far)-1]
    X_r_far = X_r_far[1:len(X_r_far)]

    baselines = np.full(len(X_l_far), baseline)

    X_near, Z_near = world_coord(baselines, f, X_l_near, X_r_near, 0)
    X_far, Z_far = world_coord(baselines, f, X_l_far, X_r_far, 0)

    print('Consecutive Images:')
    print(f'Near:')
    mean_std(Z_near)
    print(f'Far:')
    mean_std(Z_far)
    print()

if __name__ == '__main__':
    baseline = 1.625
    f = 3074

    near_1 = np.array([3384, 3032, 2640, 2280, 1896, 1480, 1100, 774, 384])
    far_1 = np.array([2976, 2816, 2628, 2464, 2276, 2060, 1896, 1732, 1568])

    near_2 = np.array([3376, 3080, 2736, 2336, 1648, 1260, 876, 500, 104])
    far_2 = np.array([2788, 2684, 2540, 2328, 2048, 1872, 1700, 1536, 1356])

    print('Day')
    reference_to_first(baseline, f, near_1, far_1)
    consecutive_images(baseline, f, near_1, far_1)

    print()

    print('Night')
    reference_to_first(baseline, f, near_2, far_2)
    consecutive_images(baseline, f, near_2, far_2)
