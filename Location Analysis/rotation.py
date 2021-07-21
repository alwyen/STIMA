import numpy as np

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
beta --> pitch
alpha --> roll
'''
def eulerAnglesZXYToRotationMatrix(gamma, beta, alpha):
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
def rotationFromWGS84GeocentricToWGS84LocalCartesian(originLatitude, originLongitude ):
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

'''
returns rotation matrix of WGS84 Geocentric frame to camera frame(?)

INPUT:
    pt      (3,1) point

Note: I understand the rotations now
    E.g. Rcb --> frame is currently in c; frame c to b, so the resulting frame is in 'b' now
         Rba --> frame is currently in b; frame b to a, so the resulting frame is in 'a' now

         Result: Rca
'''

def rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_pitch, gimbal_roll, gimbal_yaw, platform_pitch, platform_roll, platform_yaw):
    # Rotation from gimbal rotated frame (b) to camera frame (a)
    # This is a fixed, known rotation
    Rba = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # Rotation from gimbal unrotated frame (c) to gimbal rotated frame (b)
    Rcb = eulerAnglesZXYToRotationMatrix(gimbal_yaw, gimbal_pitch, gimbal_roll)
    # Rcb = eulerAnglesZYXToRotationMatrix(gimbal_yaw, gimbal_pitch, gimbal_roll)
    Rca = Rba @ Rcb

    # Rotation from platform rotated frame (d) to gimbal unrotated frame (c)
    # This is a fixed, known rotation
    Rdc = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    Rda = Rca @ Rdc

    # Rotation from platform unrotated frame (e) to platform rotated frame (d)
    Red = eulerAnglesZXYToRotationMatrix(platform_yaw, platform_pitch, platform_roll)
    # Red = eulerAnglesZYXToRotationMatrix(platform_yaw, platform_pitch, platform_roll)
    Rea = Rda @ Red

    # Rotation from WGS84 local Cartesian frame (f) to platform unrotated frame (e)
    # This is a fixed, known rotation
    Rfe = np.eye(3)
    Rfa = Rea @ Rfe

    # Rotation from WGS84 geocentric frame (g) to WGS84 local Cartesian frame (f)
    # Camera center in WGS84 geodetic coordinates
    Rgf = rotationFromWGS84GeocentricToWGS84LocalCartesian(latitude_radians, longitude_radians)
    Rga = Rfa @ Rgf

    return Rga

if __name__ == '__main__':
    pt = np.array([0,0,0]).reshape(-1,1)
    
    # # Top of phone is "top"
    # gimbal_yaw = 0
    # gimbal_pitch = -np.pi/2
    # gimbal_roll = 0

    # # Bottom of phone is "top" (i.e., upside down)
    # gimbal_yaw = np.pi
    # gimbal_pitch = -np.pi/2
    # gimbal_roll = 0

    # Right of phone is "top"
    gimbal_yaw = np.pi/2
    gimbal_pitch = -np.pi/2
    gimbal_roll = 0

    # # Left of phone is "top"
    # gimbal_yaw = -np.pi/2
    # gimbal_pitch = -np.pi/2
    # gimbal_roll = 0

    platform_yaw = 0
    platform_pitch = 0
    platform_roll = 0

    latitude = 32.87462
    longitude = -117.2258

    latitude_radians = latitude * np.pi / 180
    longitude_radians = longitude * np.pi / 180

    print(rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_yaw, gimbal_pitch, gimbal_roll, platform_yaw, platform_pitch, platform_roll))