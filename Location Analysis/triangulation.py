import numpy as np
from PIL import Image
import os

def Homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))

def Dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]

# input inhomogeneous points, return normalized, inhomogeneous x points
def normalized_x(inhomog_pts, k):
    homog_pts = Homogenize(inhomog_pts)
    norm_inhomog_pts = Dehomogenize(np.linalg.inv(k) @ homog_pts)
    return norm_inhomog_pts[0]

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

def error_stats(error_near, error_far, near_truth, far_truth):
    print(f'Average Near Error: {np.mean(error_near)}')
    print(f'Average Near % Error: {np.mean(error_near)/near_truth*100}')
    print(f'Average Far Error: {np.mean(error_far)}')
    print(f'Average Far % Error: {np.mean(error_far)/far_truth*100}')
    print()

#compare all images to first image
def reference_to_first(baseline, f, near, far, near_truth, far_truth):
    X_l_near = np.full(len(near), near[0])
    X_r_near = np.copy(near)

    X_l_far = np.full(len(far), far[0])
    X_r_far = np.copy(far)

    increments = np.arange(0,len(far),1)
    baselines = np.full(len(far), baseline)*increments


    X_near, Z_near = world_coord(baselines, f, X_l_near, X_r_near, 1)
    X_far, Z_far = world_coord(baselines, f, X_l_far, X_r_far, 1)

    # error_near = np.absolute(Z_near - near_truth)
    # error_far = np.absolute(Z_far - far_truth)

    # print('Reference to First Image:')
    # error_stats(error_near, error_far, near_truth, far_truth)

    print(f'Near:')
    # mean_std(Z_near)
    print(Z_near)
    print(f'Far:')
    # mean_std(Z_far)
    print(Z_far)
    print()

#first & second, second & third, third & fourth, ... , n & n + 1
def consecutive_images(baseline, f, near, far, near_truth, far_truth):
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

    # error_near = np.absolute(Z_near - near_truth)
    # error_far = np.absolute(Z_far - far_truth)

    # print('Consecutive Images:')
    # error_stats(error_near, error_far, near_truth, far_truth)

    print(f'Near:')
    # mean_std(Z_near)
    print(Z_near)
    print(f'Far:')
    # mean_std(Z_far)
    print(Z_far)
    print()

if __name__ == '__main__':
    # F(mm) = F(pixels) * SensorWidth(mm) / ImageWidth (pixel)
    # ground truth for near/far street lights:
    # near: ~12.5 meters
    # far: ~28.8 meters
    # probably +/- 20cm at max

    k_S9 = np.array([[3.07404255e+03, 0.00000000e+00, 1.51150000e+03], [0.00000000e+00, 3.07404255e+03, 2.01550000e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    near_truth = 12.5
    far_truth = 28.8

    baseline = 1.625
    f = 3074

    # DAY IMAGE SET
    day_near_x_1 = np.array([3376, 3036, 2644, 2280, 1896, 1484, 1104, 748, 388])
    day_near_y_1 = np.array([136, 68, 52, 48, 16, 28, 100, 64, 44])
    day_near_1 = np.vstack((day_near_x_1, day_near_y_1))

    day_far_x_1 = np.array([3028, 2872, 2696, 2520, 2332, 2116, 1944, 1792, 1624])
    day_far_y_1 = np.array([1252, 1192, 1176, 1172, 1160, 1156, 1192, 1158, 1156])
    day_far_1 = np.vstack((day_far_x_1, day_far_y_1))

    # NIGHT IMAGE SET
    night_near_x_1 = np.array([3376, 3084, 2740, 2336, 1648, 1264, 876, 500, 100])
    night_near_y_1 = np.array([376, 334, 324, 336, 312, 316, 296, 316, 288])
    night_near_1 = np.vstack((night_near_x_1,night_near_y_1))

    night_near_x_2 = np.array([3240, 2976, 2676, 2296, 1690, 1324, 964, 624, 256])
    night_near_y_2 = np.array([548, 512, 504, 512, 484, 500, 484, 504, 472])
    night_near_2 = np.vstack((night_near_x_2, night_near_y_2))

    night_far_x_1 = np.array([2784, 2684, 2540, 2328, 2048, 1972, 1700, 1536, 1356])
    night_far_y_1 = np.array([1508, 1492, 1488, 1500, 1488, 1504, 1492, 1512, 1504])
    night_far_1 = np.vstack((night_far_x_1, night_far_y_1))

    # DAY NORMALIZED
    norm_day_near_1 = normalized_x(day_near_1, k_S9)
    norm_day_far_1 = normalized_x(day_far_1, k_S9)

    # NIGHT NORMALIZED
    norm_night_near_1 = normalized_x(night_near_1, k_S9)
    norm_night_near_2 = normalized_x(night_near_2, k_S9)
    norm_night_far_1 = normalized_x(night_far_1, k_S9)

    print('Day')
    print('Reference to First:')
    reference_to_first(baseline, f, day_near_1[0], day_far_1[0], near_truth, far_truth)
    print('Consecutive')
    consecutive_images(baseline, f, day_near_1[0], day_far_1[0], near_truth, far_truth)
    
    # reference_to_first(baseline, f, norm_day_near_1, norm_day_far_1, near_truth, far_truth)
    # consecutive_images(baseline, f, norm_day_near_1, norm_day_far_1, near_truth, far_truth)

    # reference_to_first(baseline, f, day_near_2, day_far_1, near_truth, far_truth)
    # consecutive_images(baseline, f, day_near_2, day_far_1, near_truth, far_truth)

    print()

    print('Night')
    print('Reference to First')
    reference_to_first(baseline, f, night_near_1[0], night_far_1[0], near_truth, far_truth)
    print('Consecutive')
    consecutive_images(baseline, f, night_near_1[0], night_far_1[0], near_truth, far_truth)

    # reference_to_first(baseline, f, norm_night_near_1, norm_night_far_1, near_truth, far_truth)
    # consecutive_images(baseline, f, norm_night_near_1, norm_night_far_1, near_truth, far_truth)