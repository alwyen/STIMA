import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import time

def open_image(img_path):
    return np.array(Image.open(img_path), dtype='float')/255.

def Sinc(x):
    if x == 0:
        return 1
    else:
        return np.sin(x)/x

def skew(w):
    # Returns the skew-symmetrix represenation of a vector
    return np.array([[0, -w[2][0], w[1][0]], [w[2][0], 0, -w[0][0]], [-w[1][0], w[0][0], 0]])

def Deparameterize_Omega(w):
    # Deparameterizes to get rotation matrix
    mag_w = np.linalg.norm(w)
    R = np.eye(3) + Sinc(mag_w)*skew(w) + (1 - np.cos(mag_w))/(mag_w**2)*skew(w)@skew(w)
    return R

def Parameterize_Rotation(R):
    # Parameterizes rotation matrix into its axis-angle representation
    
    U, D, V_T = np.linalg.svd(R - np.eye(3))
    
    V = V_T[len(V_T)-1].reshape(-1,1)
    V_hat = np.array([R[2][1]-R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
    
    sin_theta = (V.T @ V_hat)/2
    cos_theta = (np.trace(R)-1)/2
    
    theta = math.atan2(sin_theta, cos_theta)
    
    w = theta/np.linalg.norm(V)*V
    
    if theta > np.pi:
        w = (1 - 2*np.pi/theta*np.ceil((theta - np.pi)/2/np.pi))*w

    return w, theta

def calc_camera_proj_matrix(K, R, t):
    return K @ np.hstack((R, t))

def Homogenize(x):
    return np.vstack((x,np.ones((1,x.shape[1]))))

def Dehomogenize(x):   
    return x[:-1]/x[-1]

def angle_axis_to_rotation(angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    norm_axis = axis / np.linalg.norm(axis)
    x = norm_axis[0][0]
    y = norm_axis[1][0]
    z = norm_axis[2][0]
    R = np.zeros((3,3))
    R[0][0] = t*x**2 + c
    R[0][1] = t*x*y - z*s
    R[0][2] = t*x*z + y*s
    R[1][0] = t*x*y + z*s
    R[1][1] = t*y**2 + c
    R[1][2] = t*y*z - x*s
    R[2][0] = t*x*z - y*s
    R[2][1] = t*y*z + x*s
    R[2][2] = t*z**2 + c

    return R

# calculate coefficients of X, Y, Z, and W from a 2D inhomogeneous point and the camera projection matrix
def sys_of_eq(x, P):
    x1_coeff = P[2][0]*x[0][0] - P[0][0]
    y1_coeff = P[2][1]*x[0][0] - P[0][1]
    z1_coeff = P[2][2]*x[0][0] - P[0][2]
    w1_coeff = P[2][3]*x[0][0] - P[0][3]

    x2_coeff = P[2][0]*x[1][0] - P[1][0]
    y2_coeff = P[2][1]*x[1][0] - P[1][1]
    z2_coeff = P[2][2]*x[1][0] - P[1][2]
    w2_coeff = P[2][3]*x[1][0] - P[1][3]

    row1 = np.array([x1_coeff, y1_coeff, z1_coeff, w1_coeff])
    row2 = np.array([x2_coeff, y2_coeff, z2_coeff, w2_coeff])

    return row1, row2

def interpolate_rotation(alpha, R1, R2):
    w, theta = Parameterize_Rotation(R2 @ R1.T)
    return Deparameterize_Omega(alpha * w) @ R1

def camera_translation(R1, t1, R2):
    return R2 @ (R1.T @ t1) # this should just be a 3x3 zero matrix...?

def calc_projective_transformation(K1, R1, K2, R2):
    return K2 @ R2 @ R1.T @ np.linalg.inv(K1)

def epipolar_rectification(K1, R1, t1, K2, R2, t2):
    Rinterp = interpolate_rotation(0.5, R1, R2)
    u = camera_translation(R2, t2, Rinterp) - camera_translation(R1, t1, Rinterp)
    vhat = np.array([1, 0, 0]).reshape(-1,1)
    if 0 > u.T @ vhat:
        vhat[0][0] = -1
    
    angle = np.arccos(u.T @ vhat / np.linalg.norm(u))
    axis = np.cross(u.T, vhat.T).reshape(-1,1)

    R_X = angle_axis_to_rotation(angle, axis)
    Rrectified = R_X @ Rinterp

    t1rectified = camera_translation(R1, t1, Rrectified)
    t2rectified = camera_translation(R2, t2, Rrectified)

    alpha = (K1[0][0] + K2[0][0] + K1[1][1] + K2[1][1]) / 4
    x0 = (K1[0][2] + K2[0][2]) / 2
    y0 = (K1[1][2] + K2[1][2]) / 2

    Krectified = np.zeros((3,3))

    Krectified[0][0] = alpha
    Krectified[0][2] = x0
    Krectified[1][1] = alpha
    Krectified[1][2] = y0
    Krectified[2][2] = 1

    H1 = calc_projective_transformation(K1, R1, Krectified, Rrectified)
    H2 = calc_projective_transformation(K2, R2, Krectified, Rrectified)

    return Krectified, Rrectified, t1rectified, t2rectified, H1, H2

def calc_epipole(E):
    U, S, V_T = np.linalg.svd(E)
    e = V_T[len(V_T)-1].reshape(-1,1)
    return e

'''
x1:             2D inhomog
x2:             2D inhomog
P1rectified:    rectified camera projection matrix for img1 (left camera)
P2rectified:    rectified camera projection matrix for img2 (right camera)
'''
# last row of SVD is answer(?)
def triangulation2View(x1, x2, P1rectified, P2rectified):
    row1, row2 = sys_of_eq(x1, P1rectified)
    row3, row4 = sys_of_eq(x2, P2rectified)

    A = np.vstack((row1, row2, row3, row4))

    U, S, V_T = np.linalg.svd(A)

    X = V_T[len(V_T)-1].reshape(-1,1)

    print(Dehomogenize(X))

def rectify_image(original_image, rectified_image, H, min_row_val, min_col_val):
    print(original_image.shape[0])

    print(original_image.shape[1])

    for row in range(rectified_image.shape[0]):
        for col in range(rectified_image.shape[1]):
            rec_pt = np.array([col, row]).reshape(-1,1)

            if min_row_val < 0:
                rec_pt[1][0] = rec_pt[1][0] + min_row_val
            elif min_row_val >= 0:
                rec_pt[1][0] = rec_pt[1][0] - min_row_val

            if min_col_val < 0:
                rec_pt[0][0] = rec_pt[0][0] + min_col_val
            elif min_row_val >= 0:
                rec_pt[0][0] = rec_pt[0][0] - min_col_val

            homog_rec_pt = Homogenize(rec_pt)
            img_pt = Dehomogenize(np.linalg.inv(H) @ homog_rec_pt)

            img_row = math.ceil(img_pt[1][0])
            img_col = math.ceil(img_pt[0][0])

            # print(col)
            # source_path.contains_point([xCoord, yCoord] --> WHAT
            if img_row >= 0 and img_row < original_image.shape[0] and img_col >= 0 and img_col < original_image.shape[1]:
                rectified_image[row][col] = original_image[img_row][img_col]

    return rectified_image

def epipolar_rectify_images(H1, H2, I1, I2):
    # points are defined by [x,y]!!!
    I1_top_left = np.array([0, 0]).reshape(-1,1)
    I1_top_right = np.array([I1.shape[1], 0]).reshape(-1,1)
    I1_bottom_left = np.array([0, I1.shape[0]]).reshape(-1,1)
    I1_bottom_right = np.array([I1.shape[1], I1.shape[0]]).reshape(-1,1)

    I2_top_left = np.array([0, 0]).reshape(-1,1)
    I2_top_right = np.array([I2.shape[1], 0]).reshape(-1,1)
    I2_bottom_left = np.array([0, I2.shape[0]]).reshape(-1,1)
    I2_bottom_right = np.array([I2.shape[1], I2.shape[0]]).reshape(-1,1)

    pts_I1 = np.hstack((I1_top_left, I1_top_right, I1_bottom_left, I1_bottom_right))
    pts_I2 = np.hstack((I2_top_left, I2_top_right, I2_bottom_left, I2_bottom_right))

    pts_I1_rec = Dehomogenize(H1 @ Homogenize(pts_I1))
    pts_I2_rec = Dehomogenize(H2 @ Homogenize(pts_I2))

    new_bounds = np.hstack((pts_I1_rec, pts_I2_rec))

    min_row = math.ceil(min(new_bounds[1]))
    max_row = math.ceil(max(new_bounds[1]))
    min_col = math.ceil(min(new_bounds[0]))
    max_col = math.ceil(max(new_bounds[0]))

    if min_row < 0:
        row_rec_bound = max_row + abs(min_row)
    elif min_row > 0:
        row_rec_bound = max_row - min_row
    
    if min_col < 0:
        col_rec_bound = max_col + abs(min_col)
    elif min_col > 0:
        col_rec_bound = max_col - min_col

    # DEFINE SHAPE IN TERMS OF IMAGE COORDINATES NOW
    left_rectified = np.zeros((row_rec_bound, col_rec_bound, 3))
    right_rectified = np.zeros((row_rec_bound, col_rec_bound, 3))

    print(I1.shape)

    print(left_rectified.shape)

    print(time.ctime())

    curr_time = time.time()

    left_rectified = rectify_image(I1, left_rectified, H1, min_row, min_col)
    plt.imsave('left_rectified.jpg', left_rectified)
    # plt.imsave('Ben_left_rectified.jpg', left_rectified)

    print(f'Rectification took {time.time() - curr_time} seconds')
    curr_time = time.time()

    right_rectified = rectify_image(I2, right_rectified, H2, min_row, min_col)
    plt.imsave('right_rectified.jpg', right_rectified)
    # plt.imsave('Ben_right_rectified.jpg', right_rectified)

    print(f'Rectification took {time.time() - curr_time} seconds')


if __name__ == '__main__':
    img_path_1 = 'left_led.jpg'
    img_path_2 = 'right_led.jpg'

    I1 = open_image(img_path_1)
    I2 = open_image(img_path_2)

    #left camera
    K1 = np.array([3290.62072, 0, 2006.98028, 0, 3327.56775, 1484.04255, 0, 0, 1]).reshape(3,3)
    #right camera
    K2 = np.array([3346.46355, 0, 2074.04360, 0, 3384.51059, 1445.38457, 0, 0, 1]).reshape(3,3)

    #angle of right camera
    omega = np.array([-0.02360, -0.00790, -0.02877]).reshape(-1,1)

    R1 = np.eye(3)
    R2 = Deparameterize_Omega(omega)

    t1 = np.array([0, 0, 0]).reshape(-1,1)
    t2 = np.array([ -278.75844, 10.10778, 17.38857]).reshape(-1,1)

    print(np.linalg.norm(t2))

    Krectified, Rrectified, t1rectified, t2rectified, H1, H2 = epipolar_rectification(K1, R1, t1, K2, R2, t2)

    Ben_X = np.array([4.5106863134821099e+25, 3.7251903855656536e+25, 1.0542971220170820e+26, 6.1319918022837758e+22]).reshape(-1,1)

    x1 = np.array([3195, 2662]).reshape(-1,1)
    x2 = np.array([2665, 2662]).reshape(-1,1)

    # Krectified = np.array([3337.2906524999999, 0.0000000000000000, 2040.5119399999999, \
    #                        0.0000000000000000, 3337.2906524999999, 1464.7135599999999, \
    #                        0.0000000000000000, 0.0000000000000000, 1.0000000000000000]).reshape(3,3)

    # Rrectified = np.array([0.99748768379921904, -0.0058985981993709821, -0.070594101794352840, \
    #                        0.0066944808383461512, 0.99991661688309375, 0.011042789836512600, \
    #                        0.070523078457864347, -0.011487637718467060, 0.99744399821968710]).reshape(3,3)
    
    # t1rectified = np.array([0, 0, 0]).reshape(-1,1)

    # t2rectified = np.array([-279.48308974678037, 2.3619994848900205e-14, 7.8159700933611020e-14]).reshape(-1,1)

    P1rectified = calc_camera_proj_matrix(Krectified, Rrectified, t1rectified)
    P2rectified = calc_camera_proj_matrix(Krectified, Rrectified, t2rectified)

    triangulation2View(x1, x2, P1rectified, P2rectified)
    print()
    print(Dehomogenize(Ben_X))

    Ben_H1 = np.array([1.0630228122134058, -0.014193587041567837, 43.652643864836932, \
                       0.041331132756986265, 0.99727420761593855, 81.508106854624259, \
                       2.1431542696255907e-05, -3.4522626078663791e-06, 0.95955461926217112]).reshape(3,3)
    Ben_H2 = np.array([1.0398997943316142, -0.028416211609224532, 88.682229854250352, \
                       0.065524141468807931, 0.99018383318183045, 0.050438250545255414, \
                       1.8712552518662292e-05, 2.9977058405924798e-06, 0.95484232868928220]).reshape(3,3)

    epipolar_rectify_images(H1, H2, I1, I2)
    # epipolar_rectify_images(Ben_H1, Ben_H2, I1, I2)

    '''
    # print('H1 = ')
    # print(H1)
    # print()

    # print('H2 = ')
    # print(H2)
    # print()

    # print('Krectified = ')
    # print(Krectified)
    # print()

    # print('Rrectified = ')
    # print(Rrectified)
    # print()

    # print('t1rectified = ')
    # print(t1rectified)
    # print()

    # print('t2rectified = ')
    # print(t2rectified)
    # print()

    # E = skew(t2) @ R2

    # e = calc_epipole(E.T) # left epipole
    # e_prime = calc_epipole(E) # right epipole

    # e.T @ E @ e' should be 0
    # print(left_epipole.T @ E @ right_epipole)

    # print(left_epipole)
    # print(right_epipole)
    '''