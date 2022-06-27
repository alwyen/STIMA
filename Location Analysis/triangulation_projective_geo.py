import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import time
import rotation
import glob
import os
import pandas as pd
import simplekml
from os import path

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

    return Dehomogenize(X)

'''
R1 --> np.eye(3)
omega --> R2 = Deparameterize_Omega(omega)

INPUT:
    x1                  2D inhomogeneous point of left camera
    x2                  2D inhomogeneous point of right camera
    C                   3D inhomogeneous geocentric point of the stereo camera center
    latitude_origin     latitude in degrees
    longitude_origin    longitude in degrees
    plat_yaw            yaw in degrees
    plat_pitch          pitch in degrees
    plat_roll           roll in degrees
'''
def geocentric_triangulation2View(left_img, right_img, x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12):
    # # Top of phone is "top"
    gimbal_yaw = 0
    gimbal_pitch = -np.pi/2
    gimbal_roll = 0

    # # Bottom of phone is "top" (i.e., upside down)
    # gimbal_yaw = np.pi
    # gimbal_pitch = -np.pi/2
    # gimbal_roll = 0

    # Right of phone is "top"
    # gimbal_yaw = np.pi/2
    # gimbal_pitch = -np.pi/2
    # gimbal_roll = 0

    # # Left of phone is "top"
    # gimbal_yaw = -np.pi/2
    # gimbal_pitch = -np.pi/2
    # gimbal_roll = 0

    # print('H1:')
    # print(H1)
    # print()

    # print('H2:')
    # print(H2)
    # print()

    latitude_radians = np.radians(latitude_origin)
    longitude_radians = np.radians(longitude_origin)

    plat_yaw_radians = np.radians(plat_yaw)
    plat_pitch_radians = np.radians(plat_pitch)
    plat_roll_radians = np.radians(plat_roll)

    R1_geocentric = rotation.rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_pitch, gimbal_roll, gimbal_yaw, plat_pitch_radians, plat_roll_radians, plat_yaw_radians)
    t1_geocentric = -R1_geocentric @ C
    # print('rotation from geocentric to camera frame')
    # print(R1_geocentric)
    # print

    # print()
    # print(R1_geocentric @ C)
    # print()

    R2_geocentric = R12 @ R1_geocentric
    t2_geocentric = R12 @ t1_geocentric + t12

    Krectified, Rrectified, t1rectified, t2rectified, H1, H2 = epipolar_rectification(K1, R1_geocentric, t1_geocentric, K2, R2_geocentric, t2_geocentric)
    K1rectified, K2rectified = rectified_calibration_matrices(Krectified, H1, H2, left_img, right_img)

    P1_geocentric = calc_camera_proj_matrix(K1rectified, Rrectified, t1rectified)
    P2_geocentric = calc_camera_proj_matrix(K2rectified, Rrectified, t2rectified)

    # print(np.linalg.norm(P1_geocentric))
    # print(np.linalg.norm(P2_geocentric))

    #frobius norm projection matrices
    norm_P1_geocentric = P1_geocentric / np.linalg.norm(P1_geocentric)
    norm_P2_geocentric = P2_geocentric / np.linalg.norm(P2_geocentric)

    # print('Rrectified')
    # print(Rrectified)
    # print('t1rectified')
    # print(t1rectified)
    # print('t2rectified')
    # print(t2rectified)
    # print()



    # norm_P1_geocentric = P1_geocentric
    # norm_P2_geocentric = P2_geocentric

    # K1rec_inv = np.linalg.inv(K1rectified)
    # K2rec_inv = np.linalg.inv(K2rectified)

    # x1 = Dehomogenize (K1rec_inv @ Homogenize(x1))
    # x2 = Dehomogenize (K2rec_inv @ Homogenize(x2))

    # norm_P1_geocentric = K1rec_inv @ norm_P1_geocentric
    # norm_P2_geocentric = K2rec_inv @ norm_P2_geocentric

    triangulated_geocentric_point = triangulation2View(x1, x2, norm_P1_geocentric, norm_P2_geocentric)

    return triangulated_geocentric_point

def gps_estimation(file_name, geo_centric_detic_path, xleft_path, yleft_path, xright_path, yright_path, light_gis_path, K1, K2, R12, t12, left_img, right_img):
    # geo_centric_detic unpacking
    geo_centric_detic_df = pd.read_csv(geo_centric_detic_path)
    name_list = geo_centric_detic_df.Base_Name.tolist()
    yaw_list = geo_centric_detic_df.Yaw_X.tolist()
    pitch_list = geo_centric_detic_df.Pitch_Y.tolist()
    roll_list = geo_centric_detic_df.Roll_Z.tolist()
    geo_x_list = geo_centric_detic_df.Geocentric_X.tolist()
    geo_y_list = geo_centric_detic_df.Geocentric_Y.tolist()
    geo_z_list = geo_centric_detic_df.Geocentric_Z.tolist()
    lat_list = geo_centric_detic_df.Lat.tolist()
    long_list = geo_centric_detic_df.Long.tolist()

    xleft_df = pd.read_csv(xleft_path)
    yleft_df = pd.read_csv(yleft_path)

    xright_df = pd.read_csv(xright_path)
    yright_df = pd.read_csv(yright_path)

    gis_df = pd.read_csv(light_gis_path)

    num_lights = 0
    error_list = list()

    light_number_list = list()
    est_geo_x_list = list()
    est_geo_y_list = list()
    est_geo_z_list = list()

    # temp fix
    light1_error_list = list()
    light2_error_list = list()
    light4_error_list = list()

    min_error = 9999
    max_error = -9999

    # also need to cycle through lights 1-4
    for i in range(len(name_list)):
        # left = 'left_' + name_list[i]
        # right = 'right_' + name_list[i]

        print(name_list[i])

        # cycling through lights 1-4
        for j in range(1,5):

            xleft_temp = xleft_df.loc[xleft_df['Base_Name'] == name_list[i]]
            yleft_temp = yleft_df.loc[yleft_df['Base_Name'] == name_list[i]]

            xright_temp = xright_df.loc[xright_df['Base_Name'] == name_list[i]]
            yright_temp = yright_df.loc[yright_df['Base_Name'] == name_list[i]]

            # print(xleft_temp)

            xleft_coord = xleft_temp.iloc[0][j]
            yleft_coord = yleft_temp.iloc[0][j]

            xright_coord = xright_temp.iloc[0][j]
            yright_coord = yright_temp.iloc[0][j]

            if xleft_coord == 0:
                continue

            print(f'Light {j}')

            # x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, t1, K2, t2, omega

            x1 = np.array([xleft_coord, yleft_coord]).reshape(-1,1)
            x2 = np.array([xright_coord, yright_coord]).reshape(-1,1)

            C_origin = np.array([geo_x_list[i], geo_y_list[i], geo_z_list[i]]).reshape(-1,1)

            latitude_origin = lat_list[i]
            longitude_origin = long_list[i]

            plat_yaw = yaw_list[i]
            plat_pitch = pitch_list[i]
            plat_roll = roll_list[i]

            plat_yaw = plat_yaw + 13

            estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)

            light_j = gis_df.loc[gis_df['Light_Number'] == j]

            geo_x = light_j.iloc[0]['Geocentric_X']
            geo_y = light_j.iloc[0]['Geocentric_Y']
            geo_z = light_j.iloc[0]['Geocentric_Z']

            light_number_list.append(j)
            est_geo_x_list.append(estimated_geo_point[0][0])
            est_geo_y_list.append(estimated_geo_point[1][0])
            est_geo_z_list.append(estimated_geo_point[2][0])

            light_geo_coord = np.array([geo_x, geo_y, geo_z]).reshape(-1,1)
            
            error = np.linalg.norm(estimated_geo_point - light_geo_coord)
            est_vector = estimated_geo_point - C_origin
            truth_vector = light_geo_coord - C_origin
            angle = np.degrees(rotation.angle(est_vector, truth_vector))
            # angle = np.degrees(rotation.angle(norm_est_vector, norm_truth_vector))

            error_list.append(error)
            num_lights += 1

            if j == 1:
                light1_error_list.append(error)
            elif j == 2:
                light2_error_list.append(error)
            elif j == 4:
                light4_error_list.append(error)

            print(f'Error: {error}')
            print(f'Angle Error: {angle}')

            print('estimated geocentric point')
            print(estimated_geo_point)
            print()

            if error > max_error and error < 80:
            # if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
        
        print()

        mean_error = np.mean(np.array(error_list))

    print(f'Average error for {num_lights} lights: {mean_error}')

    print(f'Min Error: {min_error}')
    print(f'Max Error: {max_error}')

    avg_error_1 = np.mean(np.array(light1_error_list))
    avg_error_2 = np.mean(np.array(light2_error_list))
    avg_error_4 = np.mean(np.array(light4_error_list))

    print(f'Average error light1: {avg_error_1}')
    print(f'Average error light2: {avg_error_2}')
    print(f'Average error light4: {avg_error_4}')

    # d = {'Light_Number':light_number_list, 'Error':error_list, 'Est_Geocentric_X':est_geo_x_list, 'Est_Geocentric_Y':est_geo_y_list, 'Est_Geocentric_Z':est_geo_z_list}
    # est_geocentric_df = pd.DataFrame(data = d)
    # # file_name = 'estimated_geocentric_coords.csv'
    # if path.exists(file_name):
    #     print('Overwriting file; press [y] to continue')
    #     user_input = input()
    #     if user_input == 'y':
    #         est_geocentric_df.to_csv(file_name)
    # else:
    #     est_geocentric_df.to_csv(file_name)

def column_error_analysis(geo_centric_detic_path, xleft_path, yleft_path, xright_path, yright_path, light_gis_path, K1, K2, R12, t12, dist_away):
    # geo_centric_detic unpacking
    geo_centric_detic_df = pd.read_csv(geo_centric_detic_path)
    name_list = geo_centric_detic_df.Base_Name.tolist()
    yaw_list = geo_centric_detic_df.Yaw_X.tolist()
    pitch_list = geo_centric_detic_df.Pitch_Y.tolist()
    roll_list = geo_centric_detic_df.Roll_Z.tolist()
    geo_x_list = geo_centric_detic_df.Geocentric_X.tolist()
    geo_y_list = geo_centric_detic_df.Geocentric_Y.tolist()
    geo_z_list = geo_centric_detic_df.Geocentric_Z.tolist()
    lat_list = geo_centric_detic_df.Lat.tolist()
    long_list = geo_centric_detic_df.Long.tolist()

    xleft_df = pd.read_csv(xleft_path)
    yleft_df = pd.read_csv(yleft_path)

    xright_df = pd.read_csv(xright_path)
    yright_df = pd.read_csv(yright_path)

    gis_df = pd.read_csv(light_gis_path)

    num_lights = 0
    error_list = list()

    light_number_list = list()
    est_geo_x_list = list()
    est_geo_y_list = list()
    est_geo_z_list = list()

    # also need to cycle through lights 1-4
    for i in range(len(name_list)):
        # left = 'left_' + name_list[i]
        # right = 'right_' + name_list[i]

        print(name_list[i])

        # cycling through lights 1-4
        for j in range(1,5):

            xleft_temp = xleft_df.loc[xleft_df['Base_Name'] == name_list[i]]
            yleft_temp = yleft_df.loc[yleft_df['Base_Name'] == name_list[i]]

            xright_temp = xright_df.loc[xright_df['Base_Name'] == name_list[i]]
            yright_temp = yright_df.loc[yright_df['Base_Name'] == name_list[i]]

            # print(xleft_temp)

            xleft_coord = xleft_temp.iloc[0][j]
            yleft_coord = yleft_temp.iloc[0][j]

            xright_coord = xright_temp.iloc[0][j]
            yright_coord = yright_temp.iloc[0][j]

            if xleft_coord == 0:
                continue

            print(f'Light {j}')

            # x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, t1, K2, t2, omega

            for k in range(-dist_away, dist_away+1):
                print(f'Shifting right column pixel by: {k}')
                temp_xright_coord = xright_coord + k
                temp_yright_coord = yright_coord + k

                x1 = np.array([xleft_coord, yleft_coord]).reshape(-1,1)
                x2 = np.array([temp_xright_coord, temp_yright_coord]).reshape(-1,1)

                C_origin = np.array([geo_x_list[i], geo_y_list[i], geo_z_list[i]]).reshape(-1,1)

                latitude_origin = lat_list[i]
                longitude_origin = long_list[i]

                plat_yaw = yaw_list[i]
                plat_pitch = pitch_list[i]
                plat_roll = roll_list[i]

                plat_yaw = plat_yaw + 13

                estimated_geo_point = geocentric_triangulation2View(x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)

                light_j = gis_df.loc[gis_df['Light_Number'] == j]

                geo_x = light_j.iloc[0]['Geocentric_X']
                geo_y = light_j.iloc[0]['Geocentric_Y']
                geo_z = light_j.iloc[0]['Geocentric_Z']

                light_number_list.append(j)
                est_geo_x_list.append(estimated_geo_point[0][0])
                est_geo_y_list.append(estimated_geo_point[1][0])
                est_geo_z_list.append(estimated_geo_point[2][0])

                light_geo_coord = np.array([geo_x, geo_y, geo_z]).reshape(-1,1)
                
                error = np.linalg.norm(estimated_geo_point - light_geo_coord)
                est_vector = estimated_geo_point - C_origin
                truth_vector = light_geo_coord - C_origin
                angle = np.degrees(rotation.angle(est_vector, truth_vector))
                # angle = np.degrees(rotation.angle(norm_est_vector, norm_truth_vector))

                print(f'Error: {error}')
                print(f'Angle: {angle}')
                print()
            print()

'''
TODO: modify this for GPS error smoothing analysis
average geocentric coordinates and see if the error is reduced through that (look at average error)
'''
def error_reduction_analysis(est_coord_path, light_gis_path):
    est_df = pd.read_csv(est_coord_path)
    gis_df = pd.read_csv(light_gis_path)
    gis_light_num = gis_df.Light_Number.tolist()

    error_list = list()

    for i in range(len(gis_light_num)):
        light_i_coord_df = gis_df.loc[gis_df['Light_Number'] == gis_light_num[i]]
        geo_x = light_i_coord_df.Geocentric_X.tolist()
        geo_y = light_i_coord_df.Geocentric_Y.tolist()
        geo_z = light_i_coord_df.Geocentric_Z.tolist()        
        truth_coord = np.array([geo_x, geo_y, geo_z]).reshape(-1,1)

        est_i_coord_df = est_df.loc[est_df['Light_Number'] == gis_light_num[i]]
        est_geo_x = est_i_coord_df.Est_Geocentric_X.tolist()
        est_geo_y = est_i_coord_df.Est_Geocentric_Y.tolist()
        est_geo_z = est_i_coord_df.Est_Geocentric_Z.tolist()

        avg_geo_x = np.mean(np.array(est_geo_x))
        avg_geo_y = np.mean(np.array(est_geo_y))
        avg_geo_z = np.mean(np.array(est_geo_z))
        avg_est_coord = np.array([avg_geo_x, avg_geo_y, avg_geo_z]).reshape(-1,1)

        error = np.linalg.norm(truth_coord - avg_est_coord)
        print(f'Light {gis_light_num[i]} Error: {error}')
        error_list.append(error)

    avg_error = np.mean(np.array(error_list))
    print(f'Avg error: {avg_error}')


'''
TODO: overlay this with GPS map...? GPD maybe too much work...
'''
def plot_coords(geo_centric_detic_path, est_coord_path, light_gis_path):
    origin_df = pd.read_csv(geo_centric_detic_path)
    est_df = pd.read_csv(est_coord_path)
    gis_df = pd.read_csv(light_gis_path)
    gis_light_num = gis_df.Light_Number.tolist()

    lat_origin_list = origin_df.Lat.tolist()
    long_origin_list = origin_df.Long.tolist()

    lat_truth_list = list()
    long_truth_list = list()

    lat_est_list = list()
    long_est_list = list()

    for i in range(len(gis_light_num)):
        light_i_coord_df = gis_df.loc[gis_df['Light_Number'] == gis_light_num[i]]
        lat_truth = light_i_coord_df.Lat.tolist()
        long_truth = light_i_coord_df.Long.tolist()

        lat_truth_list.append(lat_truth)
        long_truth_list.append(long_truth)

        est_i_coord_df = est_df.loc[est_df['Light_Number'] == gis_light_num[i]]
        lat_est = est_i_coord_df.Lat.tolist()
        long_est = est_i_coord_df.Long.tolist()

        assert len(lat_est) == len(long_est)

        for j in range(len(lat_est)):
            lat = lat_est[j]
            long = long_est[j]

            lat_est_list.append(lat)
            long_est_list.append(long)

    plt.figure(figsize = (10,7))
    plt.plot(long_origin_list, lat_origin_list, 'b*', label = 'Origin')
    plt.plot(long_truth_list, lat_truth_list, 'go', label = 'Ground Truth')
    plt.plot(long_est_list, lat_est_list, 'rx', label = 'Estmated Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Estimation Error Plots')
    plt.legend()
    plt.show()

def gps_to_kml(geo_centric_detic_path, est_coord_path, light_gis_path, origin_kml_name, est_kml_name, kml_save_path):
    origin_df = pd.read_csv(geo_centric_detic_path)
    est_df = pd.read_csv(est_coord_path)
    gis_df = pd.read_csv(light_gis_path)
    gis_light_num = gis_df.Light_Number.tolist()

    lat_origin_list = origin_df.Lat.tolist()
    long_origin_list = origin_df.Long.tolist()

    lat_est_list = est_df.Lat.tolist()
    long_est_list = est_df.Long.tolist()

    lat_gis_list = gis_df.Lat.tolist()
    long_gis_list = gis_df.Long.tolist()

    origin_kml = simplekml.Kml()
    est_kml = simplekml.Kml()
    gis_kml = simplekml.Kml()

    for i in range(len(lat_origin_list)):
        # kml.newpoint(name = str(i), coords = [(lat_origin_list[i], long_origin_list[i])])
        origin_kml.newpoint(name = str(i), coords = [(long_origin_list[i], lat_origin_list[i])])

    for i in range(len(lat_est_list)):
        est_kml.newpoint(name = str(i), coords = [(long_est_list[i], lat_est_list[i])])

    for i in range(len(lat_gis_list)):
        gis_kml.newpoint(name = str(i), coords = [(long_gis_list[i], lat_gis_list[i])])

    os.chdir(kml_save_path)

    origin_kml.save(origin_kml_name)
    est_kml.save(est_kml_name)
    gis_kml.save('gis.kml')

# def rectify_image(original_image, rectified_image, H, min_row_val, min_col_val):
def rectify_image(original_image, rectified_image, H):
    # print(original_image.shape[0])

    # print(original_image.shape[1])

    for row in range(rectified_image.shape[0]):
        for col in range(rectified_image.shape[1]):
            rec_pt = np.array([col, row]).reshape(-1,1)

            #this is the rectified point we're modifying...
            # if min_row_val < 0:
            #     rec_pt[1][0] = rec_pt[1][0] + min_row_val #adding negative value to be negative
            # elif min_row_val >= 0:
            #     rec_pt[1][0] = rec_pt[1][0] - min_row_val #I think you also need to add the value here...

            # if min_col_val < 0:
            #     rec_pt[0][0] = rec_pt[0][0] + min_col_val
            # elif min_col_val >= 0:
            #     rec_pt[0][0] = rec_pt[0][0] + min_col_val

            homog_rec_pt = Homogenize(rec_pt)
            img_pt = Dehomogenize(np.linalg.inv(H) @ homog_rec_pt)

            img_row = math.ceil(img_pt[1][0])
            img_col = math.ceil(img_pt[0][0])

            # print(col)
            # source_path.contains_point([xCoord, yCoord] --> WHAT
            if img_row >= 0 and img_row < original_image.shape[0] and img_col >= 0 and img_col < original_image.shape[1]:
                rectified_image[row][col] = original_image[img_row][img_col]

    return rectified_image

# approximately takes 10 minutes per image
# ask ben why -0.5??
def epipolar_rectify_images(H1, H2, I1, I2, name_left, name_right):
    # points are defined by [x,y]!!!
    I1_top_left = np.array([-0.5, -0.5]).reshape(-1,1)
    I1_top_right = np.array([I1.shape[1]-0.5, -0.5]).reshape(-1,1)
    I1_bottom_left = np.array([-0.5, I1.shape[0]-0.5]).reshape(-1,1)
    I1_bottom_right = np.array([I1.shape[1]-0.5, I1.shape[0]-0.5]).reshape(-1,1)

    I2_top_left = np.array([-0.5, -0.5]).reshape(-1,1)
    I2_top_right = np.array([I2.shape[1]-0.5, -0.5]).reshape(-1,1)
    I2_bottom_left = np.array([-0.5, I2.shape[0]-0.5]).reshape(-1,1)
    I2_bottom_right = np.array([I2.shape[1]-0.5, I2.shape[0]-0.5]).reshape(-1,1)

    pts_I1 = np.hstack((I1_top_left, I1_top_right, I1_bottom_left, I1_bottom_right))
    pts_I2 = np.hstack((I2_top_left, I2_top_right, I2_bottom_left, I2_bottom_right))

    pts_I1_rec = Dehomogenize(H1 @ Homogenize(pts_I1))
    pts_I2_rec = Dehomogenize(H2 @ Homogenize(pts_I2))

    min_row_I1 = math.ceil(min(pts_I1_rec[1]))
    max_row_I1 = math.ceil(max(pts_I1_rec[1]))

    min_row_I2 = math.ceil(min(pts_I2_rec[1]))
    max_row_I2 = math.ceil(max(pts_I2_rec[1]))

    min_col_I1 = math.ceil(min(pts_I1_rec[0]))
    max_col_I1 = math.ceil(max(pts_I1_rec[0]))

    min_col_I2 = math.ceil(min(pts_I2_rec[0]))
    max_col_I2 = math.ceil(max(pts_I2_rec[0]))

    T1 = np.eye(3)
    T2 = np.eye(3)

    T1[1][2] = -min(min_row_I1, min_row_I2)
    T1[0][2] = -min_col_I1 - 0.5

    T2[1][2] = -min(min_row_I1, min_row_I2)
    T2[0][2] = -min_col_I2 - 0.5

    new_H1 = T1 @ H1
    new_H2 = T2 @ H2

    # final_pts_I1_rec = Dehomogenize(new_H1 @ Homogenize(pts_I1))
    # final_pts_I2_rec = Dehomogenize(new_H2 @ Homogenize(pts_I2))

    # final_min_row_I1 = math.ceil(min(final_pts_I1_rec[1]))
    # final_max_row_I1 = math.ceil(max(final_pts_I1_rec[1]))

    # final_min_row_I2 = math.ceil(min(final_pts_I2_rec[1]))
    # final_max_row_I2 = math.ceil(max(final_pts_I2_rec[1]))

    # final_min_col_I1 = math.ceil(min(final_pts_I1_rec[0]))
    # final_max_col_I1 = math.ceil(max(final_pts_I1_rec[0]))

    # final_min_col_I2 = math.ceil(min(final_pts_I2_rec[0]))
    # final_max_col_I2 = math.ceil(max(final_pts_I2_rec[0]))

    # DEFINE SHAPE IN TERMS OF IMAGE COORDINATES NOW
    left_rectified = np.zeros((max(max_row_I1, max_row_I2) - min(min_row_I1, min_row_I2) + 1, max_col_I1 - min_col_I1 + 1, 3))
    right_rectified = np.zeros((max(max_row_I1, max_row_I2) - min(min_row_I1, min_row_I2) + 1, max_col_I2 - min_col_I2 + 1, 3))

    # print(I1.shape)

    print(left_rectified.shape)

    print(time.ctime())

    curr_time = time.time()

    left_rectified = rectify_image(I1, left_rectified, new_H1)
    plt.imsave(name_left + '.jpg', left_rectified)
    # plt.imsave('Ben_left_rectified.jpg', left_rectified)

    print(f'Rectification took {time.time() - curr_time} seconds')
    curr_time = time.time()

    print(time.ctime())

    right_rectified = rectify_image(I2, right_rectified, new_H2)
    plt.imsave(name_right + '.jpg', right_rectified)
    # plt.imsave('Ben_right_rectified.jpg', right_rectified)

    print(f'Rectification took {time.time() - curr_time} seconds')

def rectify_all_images(H1, H2, left_path, right_path, save_path):
    left_name_list = glob.glob(left_path + '\*.jpg')
    right_name_list = glob.glob(right_path + '\*.jpg')

    os.chdir(left_path)
    left_names = glob.glob('*.jpg')

    os.chdir(right_path)
    right_names = glob.glob('*.jpg')

    assert len(left_name_list) == len(right_name_list)    
    assert len(left_names) == len(right_names)
    assert len(left_name_list) == len(left_names)

    os.chdir(save_path)

    for i in range(len(left_names)):
        I1 = open_image(left_name_list[i])
        I2 = open_image(right_name_list[i])
        epipolar_rectify_images(H1, H2, I1, I2, left_names[i].split('.')[0] + '_rectified', right_names[i].split('.')[0] + '_rectified')

def rectified_calibration_matrices(Krectified, H1, H2, I1, I2):
    # points are defined by [x,y]!!!
    I1_top_left = np.array([-0.5, -0.5]).reshape(-1,1)
    I1_top_right = np.array([I1.shape[1]-0.5, -0.5]).reshape(-1,1)
    I1_bottom_left = np.array([-0.5, I1.shape[0]-0.5]).reshape(-1,1)
    I1_bottom_right = np.array([I1.shape[1]-0.5, I1.shape[0]-0.5]).reshape(-1,1)

    I2_top_left = np.array([-0.5, -0.5]).reshape(-1,1)
    I2_top_right = np.array([I2.shape[1]-0.5, -0.5]).reshape(-1,1)
    I2_bottom_left = np.array([-0.5, I2.shape[0]-0.5]).reshape(-1,1)
    I2_bottom_right = np.array([I2.shape[1]-0.5, I2.shape[0]-0.5]).reshape(-1,1)

    pts_I1 = np.hstack((I1_top_left, I1_top_right, I1_bottom_left, I1_bottom_right))
    pts_I2 = np.hstack((I2_top_left, I2_top_right, I2_bottom_left, I2_bottom_right))

    pts_I1_rec = Dehomogenize(H1 @ Homogenize(pts_I1))
    pts_I2_rec = Dehomogenize(H2 @ Homogenize(pts_I2))

    min_row_I1 = math.ceil(min(pts_I1_rec[1]))
    max_row_I1 = math.ceil(max(pts_I1_rec[1]))

    min_row_I2 = math.ceil(min(pts_I2_rec[1]))
    max_row_I2 = math.ceil(max(pts_I2_rec[1]))

    min_col_I1 = math.ceil(min(pts_I1_rec[0]))
    max_col_I1 = math.ceil(max(pts_I1_rec[0]))

    min_col_I2 = math.ceil(min(pts_I2_rec[0]))
    max_col_I2 = math.ceil(max(pts_I2_rec[0]))

    T1 = np.eye(3)
    T2 = np.eye(3)

    #if add values onto T1[1][2], answer gets closer to Ben's results
    T1[1][2] = -min(min_row_I1, min_row_I2) - 0.5
    T1[0][2] = -min_col_I1 - 0.5

    T2[1][2] = -min(min_row_I1, min_row_I2) - 0.5
    T2[0][2] = -min_col_I2 - 0.5

    # print(Krectified)

    K1rectified = T1 @ Krectified
    K2rectified = T2 @ Krectified

    # print()
    # print(K1rectified)
    # print()
    # print(K2rectified)
    # print()

    return K1rectified, K2rectified


if __name__ == '__main__':
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    # geo_centric_detic_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\1_250_exp_3200_iso_inf_focus_out.csv'
    # xleft_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\feature_coords\8_13_21\left_x.csv'
    # yleft_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\feature_coords\8_13_21\left_y.csv'
    # xright_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\feature_coords\8_13_21\right_x.csv'
    # yright_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\feature_coords\8_13_21\right_y.csv'
    # light_gis_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\ground_truth_gis_lights.csv'
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    # geo_centric_detic_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\images_10_08_21\1_350_exp_3200_iso_autofocus_out.csv'
    geo_centric_detic_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data', 'images_10_08_21', '1_350_exp_3200_iso_autofocus_nobias_out.csv')
    xleft_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data', 'feature_coords', '10_08_21', 'left_x.csv')
    yleft_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data', 'feature_coords', '10_08_21', 'left_y.csv')
    xright_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data', 'feature_coords', '10_08_21', 'right_x.csv')
    yright_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data', 'feature_coords', '10_08_21', 'right_y.csv')
    light_gis_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data', 'ground_truth_gis_lights.csv')
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    # light_gis_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\arcgis_ground_truth.csv'
    # est_coord_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\8_13_21_estimated_geocentric_coords_out.csv'
    # est_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data',\images_10_08_21\estimated_geocentric_coords_biased_out.csv'
    est_coord_path = os.path.join(os.getcwd(), 'GPS_estimation', 'gps_data', 'images_10_08_21', 'estimated_geocentric_coords_unbiased_out.csv')
    # kml_save_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\kml'
    kml_save_path = os.path.join(os.getcwd(), 'kml_original_test')
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################

    '''
    use the "Image_Name" column from geo_centric_detic_coords.csv to index the coordinates of the lamps in  
    '''

    left_img_path = os.path.join(os.getcwd(), 'GPS_estimation', 'images_8_12_21', 'Left', 'left0.jpg')
    right_img_path = os.path.join(os.getcwd(), 'GPS_estimation', 'images_8_12_21', 'Right', 'right0.jpg')

    left_img = open_image(left_img_path)
    right_img = open_image(right_img_path)

    #left camera
    # fx_left = 3306.12409
    # fy_left = 3306.12409
    # cx_left = 2080.03700
    # cy_left = 1543.19040

    fx_left = 3283.87648 
    fy_left = 3299.03024
    cx_left = 2030.36278
    cy_left = 1550.33756
    
    K1 = np.array([fx_left, 0, cx_left, 0, fy_left, cy_left, 0, 0, 1]).reshape(3,3)

    #right camera
    # fx_right = 3249.55850
    # fy_right = 3249.55850
    # cx_right = 2095.01329
    # cy_right = 1562.15979

    fx_right = 3269.07563
    fy_right = 3276.23450
    cx_right = 2086.51620
    cy_right = 1587.00571

    K2 = np.array([fx_right, 0, cx_right, 0, fy_right, cy_right, 0, 0, 1]).reshape(3,3)

    # print(K1)
    # print(K2)

    #angle of right camera
    omega = np.array([0.00659, -0.01284, -0.02433]).reshape(-1,1)

    # print(Deparameterize_Omega(omega))

    t1 = np.array([0, 0, 0]).reshape(-1,1)
    t12 = np.array([-0.33418658, 0.00541115, -0.00189281]).reshape(-1,1)
    # t2 = np.array([-0.353, 0.00541115, -0.00189281]).reshape(-1,1)


    R1 = np.eye(3)
    R12 = Deparameterize_Omega(omega)

    # est_point = geocentric_triangulation2View(x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12, left_img, right_img)
    
    # first parameter not used
    gps_estimation('8_13_21_estimated_geocentric_coords.csv', geo_centric_detic_path, xleft_coord_path, yleft_coord_path, xright_coord_path, yright_coord_path, light_gis_path, K1, K2, R12, t12, left_img, right_img)
    
    # column_error_analysis(geo_centric_detic_path, xleft_coord_path, yleft_coord_path, xright_coord_path, yright_coord_path, light_gis_path, K1, K2, R12, t12, dist_away=5)
    # error_reduction_analysis(est_coord_path, light_gis_path)
    
    # plot_coords(geo_centric_detic_path, est_coord_path, light_gis_path)
    # gps_to_kml(geo_centric_detic_path, est_coord_path, light_gis_path, 'origin.kml', 'est.kml', kml_save_path)
    # gps_to_kml(geo_centric_detic_path, est_coord_path, light_gis_path, 'origin_biased.kml', 'est_biased.kml', kml_save_path)
    # gps_to_kml(geo_centric_detic_path, est_coord_path, light_gis_path, 'origin_unbiased.kml', 'est_unbiased.kml', kml_save_path)

    # Test 1; light 1
    # x1 = np.array([1922, 355]).reshape(-1,1)
    # x2 = np.array([1870, 355]).reshape(-1,1)
    # latitude_origin = 32.87507
    # longitude_origin = -117.21848
    # plat_yaw = 355.012 - 360
    # plat_pitch = -98.61
    # plat_roll = 0.659
    # C = np.array([-2452515.379, -4768295.259, 3442350.988]).reshape(-1,1)

    # Test 2; light 1
    # x1 = np.array([420, 113]).reshape(-1,1)
    # x2 = np.array([361, 113]).reshape(-1,1)
    # latitude_origin = 32.87507
    # longitude_origin = -117.21849
    # plat_yaw = 20.928
    # plat_pitch = -98.008
    # plat_roll = 0.121
    # C = np.array([-2452516.25, -4768294.906, 3442351.042]).reshape(-1,1)

    # Test 3; light 2
    # x1 = np.array([2502, 1226]).reshape(-1,1)
    # x2 = np.array([2481, 1226]).reshape(-1,1)
    # latitude_origin = 32.87507
    # longitude_origin = -117.21849
    # plat_yaw = 20.928
    # plat_pitch = -98.008
    # plat_roll = 0.121
    # C = np.array([-2452516.25, -4768294.906, 3442351.042]).reshape(-1,1)

    # Test 4; light 2
    # x1 = np.array([1768, 1196]).reshape(-1,1)
    # x2 = np.array([1748, 1196]).reshape(-1,1)
    # latitude_origin = 32.87507
    # longitude_origin = -117.21848
    # plat_yaw = 33.766
    # plat_pitch = -98.079
    # plat_roll = -0.244
    # C = np.array([-2452515.456, -4768295.409, 3442351.096]).reshape(-1,1)

    # Test 5; light 2
    # x1 = np.array([927, 1115]).reshape(-1,1)
    # x2 = np.array([903, 1115]).reshape(-1,1)
    # latitude_origin = 32.87507
    # longitude_origin = -117.21848
    # plat_yaw = 47.323
    # plat_pitch = -97.289
    # plat_roll = -0.297
    # C = np.array([-2452515.533, -4768295.558, 3442351.205]).reshape(-1,1)

    # Test 6; light 4
    # x1 = np.array([3098, 1095]).reshape(-1,1)
    # x2 = np.array([3075, 1095]).reshape(-1,1)
    # latitude_origin = 32.87507
    # longitude_origin = -117.21848
    # plat_yaw = 47.323
    # plat_pitch = -97.289
    # plat_roll = -0.297
    # C = np.array([-2452515.533, -4768295.558, 3442351.205]).reshape(-1,1)

    # Test 7; light 4
    # x1 = np.array([2023, 1086]).reshape(-1,1)
    # x2 = np.array([2000, 1086]).reshape(-1,1)
    # latitude_origin = 32.87507
    # longitude_origin = -117.21848
    # plat_yaw = 65.984
    # plat_pitch = -96.998
    # plat_roll = -0.475
    # C = np.array([-2452515.533, -4768295.558, 3442351.205]).reshape(-1,1)

    # Test 8; light 1
    # x1 = np.array([3896, 794]).reshape(-1,1)
    # x2 = np.array([3870, 794]).reshape(-1,1)
    # latitude_origin = 32.87504
    # longitude_origin = -117.21855
    # plat_yaw = 357.191
    # plat_pitch = -105.36
    # plat_roll = -2.713
    # C = np.array([-2452525.411, -4768300.441, 3442352.97]).reshape(-1,1)

    # Test 8; light 1
    # x1 = np.array([1943, 1460]).reshape(-1,1)
    # x2 = np.array([1937, 1460]).reshape(-1,1)
    # latitude_origin = 32.87506
    # longitude_origin = -117.21848
    # plat_yaw = 41.496
    # plat_pitch = -102.063
    # plat_roll = -1.251
    # C = np.array([-2452515.616, -4768295.72, 3442350.002]).reshape(-1,1)

    # plat_yaw = plat_yaw + 13

    # est_point = geocentric_triangulation2View(x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)

    # print(est_point)

    # print(latitude_origin, longitude_origin)
    # print(C)
    # print(x1)
    # print(x2)
    # print(plat_yaw, plat_pitch, plat_roll)
    # print(K1)
    # print(t1)
    # print(K2)
    # print(t2)
    # print(omega)

    # print()

    # print(est_point)

    # # print(C)

    # print(np.linalg.norm(C - est_point))

    # gps_est_1, gps_est_2, gps_est_4, gis_light_1, gis_light_2, gis_light_4 = lights_errors(light_est_path, light_gis_gps_path)
    # error_analysis(gps_est_1, gps_est_2, gps_est_4, gis_light_1, gis_light_2, gis_light_4)

    # error = calc_dist_between_gps(32.87515225, -117.21846236, 32.87525536, -117.21840555)
    # error = calc_dist_between_gps(32.87504120, -117.21842973, 32.8750389, -117.21809992)
    # print(error)

    # x1_mid = np.array([1876, 1360]).reshape(-1,1)
    # x2_mid = np.array([1844, 1360]).reshape(-1,1)

    # x1_far = np.array([948, 1800]).reshape(-1,1)
    # x2_far = np.array([944, 1800]).reshape(-1,1)  

    # C = np.array([-2453138.6926446641, -4768009.3846089030, 3442311.5484639616]).reshape(-1,1)
    # latitude_origin = 32.87462
    # longitude_origin = -117.22580
    # plat_yaw = 23.060
    # plat_pitch = 179.728
    # plat_roll = -82.172

    # # x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, t1, K2, t2, omega
    # print('Geocentric Middle Light Coordinate:')
    # geocentric_triangulation2View(x1_mid, x2_mid, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, t1, K2, t2, omega)
    # print()
    
    # print('Geocentric Far Light Coordinate:')
    # geocentric_triangulation2View(x1_far, x2_far, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, t1, K2, t2, omega)

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################

    # img_path_1 = 'left_lights0.jpg'
    # img_path_2 = 'right_lights0.jpg'

    # I1 = open_image(img_path_1)
    # I2 = open_image(img_path_2)

    left_rectification_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\images_10_08_21\Left'
    right_rectification_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\images_10_08_21\Right'
    # save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\image_rectification'
    # save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\image_rec_8_9_21'
    save_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\image_rec_10_08_21'

    R1 = np.eye(3)
    R2 = Deparameterize_Omega(omega)

    Krectified, Rrectified, t1rectified, t2rectified, H1, H2 = epipolar_rectification(K1, R1, t1, K2, R2, t12)

    # rectify_all_images(H1, H2, left_rectification_path, right_rectification_path, save_path)

    
    # Ben_X = np.array([4.5106863134821099e+25, 3.7251903855656536e+25, 1.0542971220170820e+26, 6.1319918022837758e+22]).reshape(-1,1)

    # x1 = np.array([3195, 2662]).reshape(-1,1)
    # x2 = np.array([2665, 2662]).reshape(-1,1)

    # Krectified = np.array([3337.2906524999999, 0.0000000000000000, 2040.5119399999999, \
    #                        0.0000000000000000, 3337.2906524999999, 1464.7135599999999, \
    #                        0.0000000000000000, 0.0000000000000000, 1.0000000000000000]).reshape(3,3)

    # Rrectified = np.array([0.99748768379921904, -0.0058985981993709821, -0.070594101794352840, \
    #                        0.0066944808383461512, 0.99991661688309375, 0.011042789836512600, \
    #                        0.070523078457864347, -0.011487637718467060, 0.99744399821968710]).reshape(3,3)
    
    # t1rectified = np.array([0, 0, 0]).reshape(-1,1)

    # t2rectified = np.array([-279.48308974678037, 2.3619994848900205e-14, 7.8159700933611020e-14]).reshape(-1,1)

    # P1rectified = calc_camera_proj_matrix(Krectified, Rrectified, t1rectified)
    # P2rectified = calc_camera_proj_matrix(Krectified, Rrectified, t2rectified)

    # print(triangulation2View(x1, x2, P1rectified, P2rectified))
    # print()
    # print(Dehomogenize(Ben_X))

    # Ben_H1 = np.array([1.0630228122134058, -0.014193587041567837, 43.652643864836932, \
    #                    0.041331132756986265, 0.99727420761593855, 81.508106854624259, \
    #                    2.1431542696255907e-05, -3.4522626078663791e-06, 0.95955461926217112]).reshape(3,3)
    # Ben_H2 = np.array([1.0398997943316142, -0.028416211609224532, 88.682229854250352, \
    #                    0.065524141468807931, 0.99018383318183045, 0.050438250545255414, \
    #                    1.8712552518662292e-05, 2.9977058405924798e-06, 0.95484232868928220]).reshape(3,3)
    

    '''
    # 
    # headergeo_centric_detic_df = pd.read_csv(csv_path)_list = list(geo_centric_detic_df.columns)[0].split('\t')
    # geo_list = geo_centric_detic_df.values.tolist()
    # name_list = list()
    # yaw_list = list()
    # pitch_list = list()
    # roll_list = list()
    # geo_x_list = list()
    # geo_y_list = list()
    # geo_z_list = list()
    # lat_list = list()
    # long_list = list()
    # height_list = list()
    # for i in range(len(geo_list)):
    #     row = list(geo_list[i])[0].split('\t')
    #     name_list.append(row[0])
    #     yaw_list.append(float(row[1]))
    #     pitch_list.append(float(row[2]))
    #     roll_list.append(float(row[3]))
    #     geo_x_list.append(float(row[4]))
    #     geo_y_list.append(float(row[5]))
    #     geo_z_list.append(float(row[6]))
    #     lat_list.append(float(row[7]))
    #     long_list.append(float(row[8]))
    #     height_list.append(float(row[9]))

    # d = {header_list[0]: name_list, header_list[1]: yaw_list, header_list[2]: pitch_list, header_list[3]: roll_list, header_list[4]: geo_x_list, header_list[5]: geo_y_list, header_list[6]: geo_z_list, header_list[7]: lat_list, header_list[8]: long_list, header_list[9]: height_list}
    # df = pd.DataFrame(data = d)
    # df.to_csv('please_work.csv')  
    '''

    # H1 = np.array([[ 9.91275333e-01, -6.90187277e-03, 7.53407792e+01], [ 2.19327103e-03,  9.94841868e-01,  2.93493660e+01], [-2.58012514e-06, -3.24156793e-07, 1.00576887e+00]]).reshape(3,3)
    # H2 = np.array([[ 1.00263463e+00, -2.33391941e-02, 1.83809682e+01], [ 2.33640382e-02,  1.00443599e+00, -8.17564452e+01], [-4.31016659e-07, 3.40347247e-07, 1.00036065e+00]]).reshape(3,3)

    # pt1 = np.array([1428, 1292]).reshape(-1,1)
    # pt2 = np.array([1356, 1292]).reshape(-1,1)

    # orig_pt1 = Dehomogenize(np.linalg.inv(H1) @ Homogenize(pt1))
    # orig_pt2 = Dehomogenize(np.linalg.inv(H2) @ Homogenize(pt2))

    # buffer = np.array([54, 81]).reshape(-1,1)

    # print(orig_pt1-buffer)
    # print(orig_pt2-buffer)

    '''
    fx_left = 3290.62072
    fy_left = 3327.56775
    cx_left = 2006.98028
    cy_left = 1484.04255
    
    K1 = np.array([fx_left, 0, cx_left, 0, fy_left, cy_left, 0, 0, 1]).reshape(3,3)

    fx_right = 3346.46355
    fy_right = 3384.51059
    cx_right = 2074.04360
    cy_right = 1445.38457

    K2 = np.array([fx_right, 0, cx_right, 0, fy_right, cy_right, 0, 0, 1]).reshape(3,3)

    omega = np.array([ -0.02360, -0.00790, -0.02877]).reshape(-1,1)

    t1 = np.array([0, 0, 0]).reshape(-1,1)
    t2 = np.array([-278.75844, 10.10778, 17.38857]).reshape(-1,1)

    R1 = np.eye(3)
    R2 = Deparameterize_Omega(omega)
    Krectified, Rrectified, t1rectified, t2rectified, H1, H2 = epipolar_rectification(K1, R1, t1, K2, R2, t2)

    print(Krectified)
    print(Rrectified)
    print(t1rectified)
    print(t2rectified)

    # I1 = open_image('left_led.jpg')
    # I2 = open_image('right_led.jpg')
    # epipolar_rectify_images(H1, H2, I1, I2, 'right_rectified', 'left_rectified')
    '''
    