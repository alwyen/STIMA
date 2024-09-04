import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
import pandas as pd
import scipy.io as sio

### GLOBAL INITIALIZERS
filename = "C:\\Users\\Anthony\\Downloads\\calibrationData_640_480.mat"
matlab_file = sio.loadmat(filename)

intrinsicsData = []
distortionData = []

for i in range(len(matlab_file['calibration'][0][0]) - 1):
    if i < 2:
        intrinsicsData.append(matlab_file['calibration'][0][0][i])
    elif i < 4:
        distortionData.append(matlab_file['calibration'][0][0][i])
    else:
        R = matlab_file['calibration'][0][0][i]
        T = matlab_file['calibration'][0][0][i+1]
        extrinsicsData = np.hstack((R, T.T))

CAM_LEFT = intrinsicsData[0]
CAM_RIGHT = intrinsicsData[1]
DIST_LEFT = distortionData[0]
DIST_RIGHT = distortionData[1]
extrinsics = extrinsicsData
R, T = extrinsics[:,:3], extrinsics[:,3]
T = T/100 # Meter calibrated
EXTRINSICS = np.hstack((R, np.vstack(T)))


## READERS ########################

def camera_params_reader(file_path):
    with open(file_path) as file:
        lines = [line.rstrip().split(",") for line in file]

    camera_param = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    for data in lines:
        if "r1" in data:
            continue
        cam_line = np.array([[float(data[0]),
                              float(data[1]),
                              float(data[2]),
                              float(data[3]),
                              float(data[4]),
                              float(data[5])]])
        camera_param = np.vstack((camera_param, cam_line))

    return camera_param[1:,:]

def points_3d_reader(file_path):
    with open(file_path) as file:
        lines = [line.rstrip().split(",") for line in file]
    points_3d_param = np.array([[0.0, 0.0, 0.0]])

    for data in lines:
        if "x" in data:
            continue
        point_line = np.array([[float(data[0]),
                              float(data[1]),
                              float(data[2])]])
        points_3d_param = np.vstack((points_3d_param, point_line))


    return points_3d_param[1:,:]

def camera_ind_reader(file_path):
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]

    temp_data = []
    for data in lines:
        if "E1" in data:
            continue
        temp_data.append(int(data))
    camera_ind_data = np.array(temp_data)

    return camera_ind_data


def point_ind_reader(file_path):
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]

    temp_data = []
    for data in lines:
        if "E1" in data:
            continue
        temp_data.append(int(data))
    point_ind_data = np.array(temp_data)

    return point_ind_data

def points_2d_reader(file_path):
    with open(file_path) as file:
        lines = [line.rstrip().split(",") for line in file]

    points_2d_param = np.array([[0.0, 0.0, 0.0, 0.0]])
    for data in lines:
        if "x1" in data:
            continue
        point_line = np.array([[float(data[0]),
                              float(data[1]),
                              float(data[2]),
                              float(data[3])]])
        points_2d_param = np.vstack((points_2d_param, point_line))


    return points_2d_param[1:,:]

###################################
# HELPERS

def covert_to_quat(r_Matrix):
    trace = np.trace(r_Matrix)

    m00 = r_Matrix[0,0]
    m01 = r_Matrix[0,1]
    m02 = r_Matrix[0,2]
    m10 = r_Matrix[1,0]
    m11 = r_Matrix[1,1]
    m12 = r_Matrix[1,2]
    m20 = r_Matrix[2,0]
    m21 = r_Matrix[2,1]
    m22 = r_Matrix[2,2]

    if (0 < trace):
        a = np.sqrt(trace + 1)
        b = 1 / (2 * a)

        qw = a / 2
        qx = (m21 - m12) * b
        qy = (m02 - m20) * b
        qz = (m10 - m01) * b

    else:
        i = 0
        if (m11 > m00): i = 1
        if (m22 > r_Matrix[i,i]): i = 2
        

        j = (i + 1) % 3
        k = (j + 1) % 3

        a = np.sqrt(max(0, r_Matrix[i,i] - r_Matrix[j,j] - r_Matrix[k,k] + 1))
        b = 1 / (2 * a)
 
        qw = (r_Matrix[k,j] - r_Matrix[j,k]) * b
        qx = a / 2
        qy = (r_Matrix[j,i] + r_Matrix[i,j]) * b
        qz = (r_Matrix[k,i] + r_Matrix[i,k]) * b
    # qw = np.sqrt(1 + m00 + m11 + m22) /2
    # qx = (m21 - m12)/( 4 *qw)
    # qy = (m02 - m20)/( 4 *qw)
    # qz = (m10 - m01)/( 4 *qw)

    return [qw, qx, qy, qz]



###################################




def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""

    ws = camera_params[:, :3]
    trans = camera_params[:, 3:]


    R12_static = EXTRINSICS[:, :3]

    T12_static = EXTRINSICS[:, 3:].reshape((3,1))
    KL, KR = CAM_LEFT, CAM_RIGHT

    points_proj = np.array([[0,0,0,0]])
    for par in range(ws.shape[0]):
      w = ws[par,:]
      w = w.reshape((3,1))

      Rot_L = cv.Rodrigues(w)[0]
      Trans_L  = trans[par,:].reshape((3,1))

      Rot_R = R12_static @ Rot_L
      Trans_R = R12_static @ Trans_L + T12_static

      x_left = cv.projectPoints(points[par, :], Rot_L, Trans_L, KL, DIST_LEFT)[0][0]
      x_right = cv.projectPoints(points[par, :], Rot_R, Trans_R, KR, DIST_RIGHT)[0][0]

      estM = np.hstack((x_left, x_right))
      points_proj = np.vstack((points_proj, estM))


    return points_proj[1:, :]

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    #print(points_proj.shape)
    return (np.linalg.norm(points_proj - points_2d,axis=1)).ravel()#(points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size #* 4
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    #print(m)
    #print(len(i))
    for s in range(9):
        # A[4 * i, camera_indices * 6 + s] = 1
        # A[4 * i + 1, camera_indices * 6 + s] = 1
        A[i, camera_indices * 6 + s] = 1
        # A[i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        # A[4 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        # A[4 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
        A[i, n_cameras * 6 + point_indices * 3 + s] = 1
        # A[i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def main():
    exp_num = sys.argv[1]
    exp_ver = sys.argv[2]
    path = "C:\\Users\\Anthony\\Downloads\\"
    unopt_file = f"unoptimzed_nighttime_exp{exp_num}_{exp_ver}_scene_"

    camera_params_file = path + unopt_file + "camera_params.csv"
    points_3d_file = path + unopt_file + "3d_point_file.csv"
    camera_ind_file = path + unopt_file + "camera_ind_file.csv"
    point_ind_file = path + unopt_file + "point_ind.csv"
    points_2d_file = path + unopt_file + "points_2d.csv"
    frame_used_file = path + f"frames_used_exp_{exp_num}_{exp_ver}"

    camera_params = camera_params_reader(camera_params_file)
    points_3d = points_3d_reader(points_3d_file)
    camera_ind = camera_ind_reader(camera_ind_file)
    point_ind = point_ind_reader(point_ind_file)
    points_2d = points_2d_reader(points_2d_file)

    # Shape verification
    print(camera_params.shape)
    print(points_3d.shape)
    print(camera_ind.shape)
    print(point_ind.shape)
    print(points_2d.shape)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 6 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_ind, point_ind, points_2d)

    A = bundle_adjustment_sparsity(camera_params.shape[0], points_3d.shape[0], camera_ind, point_ind)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(camera_params.shape[0], n_points, camera_ind, point_ind, points_2d))
    t1 = time.time()

    frames_used = []   

    with open(frame_used_file) as file:
        lines = [line.rstrip().split(",") for line in file]
    
    for line in lines:
        frames_used.append(line)
    # if int(exp_num) == 10:
    #     # Exp 10
    #     notToUse = [393, 394, 396, 398, 399]

    #     for i in range(297,440):
    #         if i in notToUse:
    #             continue
                
    #         frames_used.append(f'frame{i}.png')
    # elif int(exp_num) == 11:
    #     for i in range(190,330):
    #         frames_used.append(f'frame{i}.png')
    # elif int(exp_num) == 12:
    #     for i in range(195,302):
    #         frames_used.append(f'frame{i}.png')
    
    camera_data_df = pd.read_csv(path + f"frame_time_stamps{exp_num}.csv")
    gps_data_df = pd.read_csv(path + f"gps_time_stamps{exp_num}.csv")
    imu_data_df = pd.read_csv(path + f"orientation_time_stamps{exp_num}.csv")


    frames_to_use = camera_data_df['frame_num'].to_list()
    frame_idx = {}
    for frame in frames_to_use:
        full_frame = frame + ".png"
        if full_frame in frames_used:
            frame_idx[full_frame[:-4]] = frames_used.index(full_frame)

    print(frame_idx)

    cam_poses = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    # points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
    # cam_poses = camera_params

    f_cam_poses = open(f'cam_poses_optimal_{exp_num}.csv', 'w')
    top_line = "timestamp,frame,"
    for i in range(0,4):
        top_line += "q" + str(i) + ","
    for i in range(0,3):
        top_line += "t" + str(i) + ","
    f_cam_poses.write(top_line[:-1] + '\n')
    
    index_to_use = []
    for index, row in camera_data_df.iterrows():
        temp_strL = ""
        #print(row['frame_num'])
        if row['frame_num'] in frame_idx:
            #print("In here")
            index_to_use.append(index)
            idx = frame_idx[row['frame_num']]
            timestamp = row['timestamp']
            frame = row['frame_num']
            rotL = cv.Rodrigues(cam_poses[idx, :3])[0]

            transL = cam_poses[idx, 3:].reshape((3,1))

            #cam_poseL = np.hstack((rotL, transL))

            quaternion = covert_to_quat(rotL)

            top_line = f"{timestamp},{frame},"
            for i in range(0,4):
                top_line += str(quaternion[i]) + ","
            for i in range(0,3):
                top_line += str(transL[i,0]) + ","
            f_cam_poses.write(top_line[:-1] + '\n')

    f_gps_pos = open(f'gps_pos_opt_{exp_num}.csv', 'w')
    top_line = "timestamp,longitude,latitude,altitude"
    f_gps_pos.write(top_line + '\n')

    #for index, row in gps_data_df.iterrows():

    for index in index_to_use:
        if index % 2 == 0:
            idx = int(index/2)
            row = gps_data_df.iloc[idx]
            timestamp = row["timestamp"]
            latitude = row["latitude"]
            longitude = row["longitude"]
            altitude = row["altitude"]
            f_gps_pos.write(f"{timestamp},{longitude},{latitude},{altitude}\n")

    f_imu_poses = open(f'imu_orientation_opt_{exp_num}.csv', 'w')
    top_line = "timestamp,"
    for i in range(0,4):
        top_line += "q" + str(i) + ","
    f_imu_poses.write(top_line[:-1] + '\n')
    
    for index, row in imu_data_df.iterrows():
        if index in index_to_use:
            timestamp = row["timestamp"]
            q0 = row["q0"]
            q1 = row["q1"]
            q2 = row["q2"]
            q3 = row["q3"]
            f_imu_poses.write(f"{timestamp},{q0},{q1},{q2},{q3}\n")
    
main()