import math
import numpy as np
import pandas as pd
import glob
import os

'''
longitude   --> x-values (-180 to 180)
latitude    --> y-values (-90 to 90)

coordinate  --> (x, y) --> (longitude, latitude)

EPSG 4326       curved surface (ellipsoid??)
EPSG 3857       flat surface (orthometric??)
from_crs(a, b)  convert from 'a' to 'b'
'''

def orthometricToEllipsoid():
    pass

'''
gamma --> yaw
alpha --> roll
beta --> pitch
'''
def eulerAnglesZXYToRotationMatrix(gamma, alpha, beta):
    sin_gamma = np.sin(gamma)
    cos_gamma = np.cos(gamma)
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    sin_beta  = np.sin(beta)
    cos_beta  = np.cos(beta)

    R = np.zeros((3,3))
    R[0][0] = cos_beta * cos_gamma - sin_alpha * sin_beta * sin_gamma
    R[0][1] = -cos_alpha * sin_gamma
    R[0][2] = cos_gamma * sin_beta + cos_beta * sin_alpha * sin_gamma
    R[1][0] = cos_beta * sin_gamma + cos_gamma * sin_alpha * sin_beta
    R[1][1] = cos_alpha * cos_gamma
    R[1][2] = sin_beta * sin_gamma - cos_beta * cos_gamma * sin_alpha
    R[2][0] = -cos_alpha * sin_beta
    R[2][1] = sin_alpha
    R[2][2] = cos_alpha * cos_beta
    return R

def eulerAnglesZYXToRotationMatrix( gamma, beta, alpha):
     sin_gamma = np.sin(gamma)
     cos_gamma = np.cos(gamma)
     sin_beta  = np.sin(beta)
     cos_beta  = np.cos(beta)
     sin_alpha = np.sin(alpha)
     cos_alpha = np.cos(alpha)

     R = np.zeros((3,3))
     R[0][0] = cos_beta * cos_gamma
     R[0][1] = sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma
     R[0][2] = cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma
     R[1][0] = cos_beta * sin_gamma
     R[1][1] = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
     R[1][2] = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma
     R[2][0] = -sin_beta
     R[2][1] = cos_beta * sin_alpha
     R[2][2] = cos_beta * cos_alpha
     return R

'''
latitude and longitude are both in radians
'''
def rotationFromWGS84GeocentricToWGS84LocalCartesian(originLatitude, originLongitude):
    sin_lat = np.sin(originLatitude)
    cos_lat = np.cos(originLatitude)
    sin_lon = np.sin(originLongitude)
    cos_lon = np.cos(originLongitude)

    R = np.zeros((3,3))
    R[0][0] = -sin_lon
    R[0][1] =  cos_lon
    R[0][2] =  0
    R[1][0] = -sin_lat * cos_lon
    R[1][1] = -sin_lat * sin_lon
    R[1][2] =  cos_lat
    R[2][0] =  cos_lat * cos_lon
    R[2][1] =  cos_lat * sin_lon
    R[2][2] =  sin_lat

    return R

# angle in radians
def rotX(theta):
    R = np.eye(3)
    R[1][1] = np.cos(theta)
    R[1][2] = -np.sin(theta)
    R[2][1] = np.sin(theta)
    R[2][2] = np.cos(theta)
    return R

def rotY(theta):
    R = np.eye(3)
    R[0][0] = np.cos(theta)
    R[0][2] = np.sin(theta)
    R[2][0] = -np.sin(theta)
    R[2][2] = np.cos(theta)
    return R

def rotZ(theta):
    R = np.eye(3)
    R[0][0] = np.cos(theta)
    R[0][1] = -np.sin(theta)
    R[1][0] = np.sin(theta)
    R[1][1] = np.cos(theta)
    return R

'''
returns rotation matrix of WGS84 Geocentric frame to camera frame(?)

INPUT:
    pt      (3,1) point

Note (1): I understand the rotations now
    E.g. Rcb --> frame is currently in c; frame c to b, so the resulting frame is in 'b' now
         Rba --> frame is currently in b; frame b to a, so the resulting frame is in 'a' now

         Result: Rca

Note (2): From "IMU+GPS-Stream" app:
        X --> Yaw
        Y --> Pitch
        Z --> Roll

Note (3):
        All angles (including platform angles) should be in radians
'''

def rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_pitch, gimbal_roll, gimbal_yaw, platform_pitch, platform_roll, platform_yaw):
    # Rotation from gimbal rotated frame (b) to camera frame (a)
    # This is a fixed, known rotation
    Rba = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # Rotation from gimbal unrotated frame (c) to gimbal rotated frame (b)
    # Rcb = eulerAnglesZXYToRotationMatrix(gimbal_yaw, gimbal_pitch, gimbal_roll).T
    Rcb = eulerAnglesZYXToRotationMatrix(gimbal_yaw, gimbal_pitch, gimbal_roll).T
    Rca = Rba @ Rcb

    # print('Rcb:')
    # print(Rcb)
    # print()

    # Rotation from platform rotated frame (d) to gimbal unrotated frame (c)
    # This is a fixed, known rotation
    Rdc = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    Rda = Rca @ Rdc

    # Rotation from platform unrotated frame (e) to platform rotated frame (d)

    Red = rotY(platform_roll) @ rotX(platform_pitch) @ rotZ(platform_yaw)
    # Red = eulerAnglesZXYToRotationMatrix(platform_yaw, platform_pitch, platform_roll).T
    # Red = rotZ(platform_yaw) @ rotX(platform_pitch) @ rotY(platform_roll)

    # print('Red:')
    # print(Red)

    Rea = Rda @ Red

    # print('Red:')
    # print(Red)
    # print()

    # Rotation from WGS84 local Cartesian frame (f) to platform unrotated frame (e)
    # This is a fixed, known rotation
    Rfe = np.eye(3)
    Rfa = Rea @ Rfe

    # Rotation from WGS84 geocentric frame (g) to WGS84 local Cartesian frame (f)
    # Camera center in WGS84 geodetic coordinates
    Rgf = rotationFromWGS84GeocentricToWGS84LocalCartesian(latitude_radians, longitude_radians)
    Rga = Rfa @ Rgf

    # print('Rgf:')
    # print(Rgf)
    # print()

    # print('Rga:')
    # print(Rga)

    return Rga


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

# https://geographiclib.sourceforge.io/cgi-bin/GeoidEval
def rotation_check(rotation_csv_path):
    rotation_df = pd.read_csv(rotation_csv_path)

    geo_x_list = rotation_df.Geocentric_X.tolist()
    geo_y_list = rotation_df.Geocentric_Y.tolist()
    geo_z_list = rotation_df.Geocentric_Z.tolist()

    # gimbal_list = rotation_df.Gimbal.tolist()
    direction_list = rotation_df.Direction.tolist()
    
    long_list = rotation_df.Long.tolist()
    lat_list = rotation_df.Lat.tolist()
    plat_yaw_list = rotation_df.Yaw_X.tolist()
    plat_pitch_list = rotation_df.Pitch_Y.tolist()
    plat_roll_list = rotation_df.Roll_Z.tolist()

    X_x = list()
    X_y = list()
    X_z = list()

    Y_x = list()
    Y_y = list()
    Y_z = list()

    Z_x = list()
    Z_y = list()
    Z_z = list()

    gimbal = list()

    for i in range(len(direction_list)):
    # for i in range(0, 3):
        # if gimbal_list[i] == 'Left':
        #     gimbal_yaw = -np.pi/2
        #     gimbal_pitch = -np.pi/2
        #     gimbal_roll = 0
        # elif gimbal_list[i] == 'Right':
        #     gimbal_yaw = np.pi/2
        #     gimbal_pitch = -np.pi/2
        #     gimbal_roll = 0
        # if gimbal_list[i] == 'Top':
        #     gimbal_yaw = 0
        #     gimbal_pitch = -np.pi/2
        #     gimbal_roll = 0
        # else:
        #     continue

        # "Top" is top orientation for 'top_orientation_test_8_4_21.csv'
        gimbal_yaw = 0
        gimbal_pitch = -np.pi/2
        gimbal_roll = 0

        latitude_radians = np.radians(lat_list[i])
        longitude_radians = np.radians(long_list[i])
        platform_pitch = np.radians(plat_pitch_list[i])
        platform_roll = np.radians(plat_roll_list[i])
        platform_yaw = np.radians(plat_yaw_list[i])

        # print(platform_yaw)
        
        platform_pitch = 0
        platform_roll = 0
        platform_yaw = 0

        R = rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_pitch, gimbal_roll, gimbal_yaw, platform_pitch, platform_roll, platform_yaw)
        C = np.array([geo_x_list[i], geo_y_list[i], geo_z_list[i]]).reshape(-1,1)


        # x_vector = (np.array([-2452517.028, -4768301.126, 3442337.549]).reshape(-1,1) - C)
        # z_vector = np.array([-2452504.123, -4768300.588, 3442347.422]).reshape(-1,1) - C

        # x_vector = x_vector / np.linalg.norm(x_vector)
        # z_vector = z_vector / np.linalg.norm(z_vector)

        # print(np.degrees(angle(x_vector, z_vector)))

        # print(R)
        # print()

        # X_10 = R[0] @ C * 10
        # Y_10 = R[1] @ C * 10
        # Z_10 = R[2] @ C * 10      

        gimbal.append(direction_list[i])

        X_10 = (R[0].reshape(-1,1) * 10) + C
        Y_10 = (R[1].reshape(-1,1) * 10) + C
        Z_10 = (R[2].reshape(-1,1) * 10) + C

        # X_10 = -(R[:,0] * 10) + C
        # Y_10 = -(R[:,1] * 10) + C
        # Z_10 = -(R[:,2] * 10) + C

        X_x.append(X_10[0][0])
        X_y.append(X_10[1][0])
        X_z.append(X_10[2][0])

        Y_x.append(Y_10[0][0])
        Y_y.append(Y_10[1][0])
        Y_z.append(Y_10[2][0])

        Z_x.append(Z_10[0][0])
        Z_y.append(Z_10[1][0])
        Z_z.append(Z_10[2][0])

        # radius_C = np.linalg.norm(C)
        # radius_Y = np.linalg.norm(Y_10)

        # if radius_C - radius_Y < 0:
        #     print('Y pointing downwards') # but not by 10 meters; is that wrong then...?
        #     print(radius_C - radius_Y)
        #     # print(np.linalg.norm(Y_10 - C))
        # elif radius_C - radius_Y > 0:
        #     print('Y pointing up')
        #     print(radius_C - radius_Y)
        #     # print(np.linalg.norm(Y_10 - C))

        # print(angle(R[0], R[1]))
        # print(angle(R[0], R[2]))
        # print(angle(R[1], R[2]))

        print(direction_list[i])
        print('X')
        print(X_10[0][0])
        print(X_10[1][0])
        print(X_10[2][0])
        print('Y')
        print(Y_10[0][0])
        print(Y_10[1][0])
        print(Y_10[2][0])
        print('Z')
        print(Z_10[0][0])
        print(Z_10[1][0])
        print(Z_10[2][0])
        
        # print(f'Angle X: {np.degrees(angle(x_vector, R[0]))}')
        # print(f'Angle Z: {np.degrees(angle(z_vector, R[2]))}')
        # print(f'Angle X: {angle(x_vector, R[0].reshape(-1,1)+C)}')
        # print(f'Angle Z: {angle(z_vector, R[2].reshape(-1,1)+C)}')

        print()

        break


    # print('X')
    # # print(f'X_x mean: {np.mean(X_x)}')
    # print(f'X_x STD: {np.std(X_x)}')
    # # print(f'X_y mean: {np.mean(X_y)}')
    # print(f'X_y STD: {np.std(X_y)}')
    # # print(f'X_z mean: {np.mean(X_z)}')
    # print(f'X_z STD: {np.std(X_z)}')
    # print()

    # print('Y')
    # # print(f'Y_x mean: {np.mean(Y_x)}')
    # print(f'Y_x STD: {np.std(Y_x)}')
    # # print(f'Y_y mean: {np.mean(Y_y)}')
    # print(f'Y_y STD: {np.std(Y_y)}')
    # # print(f'Y_z mean: {np.mean(Y_z)}')
    # print(f'Y_z STD: {np.std(Y_z)}')
    # print()

    # print('Z')
    # # print(f'Z_x mean: {np.mean(Z_x)}')
    # print(f'Z_x STD: {np.std(Z_x)}')
    # # print(f'Z_y mean: {np.mean(Z_y)}')
    # print(f'Z_y STD: {np.std(Z_y)}')
    # # print(f'Z_z mean: {np.mean(Z_z)}')
    # print(f'Z_z STD: {np.std(Z_z)}')
    # print()

        # d = {'Gimbal_Orientation':gimbal, 'X_x':X_x, 'X_y':X_y, 'X_z':X_z, 'Z_x':Z_x, 'Z_y':Z_y, 'Z_z':Z_z}
        # df = pd.DataFrame(data = d)
        # df.to_csv('top_test.csv')

def combine_test_files(path):
    os.chdir(path)
    file_list = glob.glob('*csv')
    df = pd.read_csv(file_list[0])

    for i in range(1, len(file_list)):
        temp_df = pd.read_csv(file_list[i])
        df = df.append(temp_df)

    df.to_csv('direction_check.csv')

if __name__ == '__main__':
    # test_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\STIMA\Location Analysis\rotation_test.csv'
    test_path2 = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\top_orientation_test_8_4_21.csv'
    test_path3 = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\top_orientation_test_8_9_21.csv'

    pt = np.array([0,0,0]).reshape(-1,1)
    
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

    platform_yaw = 76.848
    platform_pitch = -90.643
    platform_roll = 0.294

    latitude = 32.87505
    longitude = -117.21848

    latitude_radians = np.radians(latitude)
    longitude_radians = np.radians(longitude)

    plat_yaw_radians = np.radians(platform_yaw)
    plat_pitch_radians = np.radians(platform_pitch)
    plat_roll_radians = np.radians(platform_roll)

    # rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_pitch, gimbal_roll, gimbal_yaw, plat_pitch_radians, plat_roll_radians, plat_yaw_radians)

    rotation_check(test_path3)

    # path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\rotation_test'
    # combine_test_files(path)