import sys
sys.path.append('C:\\Users\\Anthony\\Documents\\Projects\\LightsCameraGrid\\Exp_Analysis\\mobile_analysis')
import cv2 as cv
import glob
from calibration_reader import *
import numpy as np
from scipy import linalg
import yaml
import os

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}


#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix, npz_file_name, directory):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = os.listdir(images_prefix)
    # print(images_names)
    #read all frames
    images = [cv.imread(os.path.join(images_prefix, imname), 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    flags = (cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY)
    for i, frame in enumerate(images):
        print(f'Image {i+1}/{len(images)}')
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCornersSB(gray, (rows, columns), flags)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (9, 9)#(11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join(directory, npz_file_name + '_intrinsics')
    np.savez(out_filename, intrinsic=cmtx, dist=dist, rot=rvecs, trans=tvecs)


#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1, R_est, T_est):

    assert len(frames_prefix_c0) == len(frames_prefix_c1)

    image_counter = 1
    total_images = len(frames_prefix_c0)

    #read the synched frames
    c0_images_names = sorted(frames_prefix_c0)
    c1_images_names = sorted(frames_prefix_c1)

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    flags = (cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY)
    for frame0, frame1 in zip(c0_images, c1_images):
        print(f'Pair {image_counter}/{total_images}')
        image_counter += 1

        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCornersSB(gray1, (rows, columns), flags)
        c_ret2, corners2 = cv.findChessboardCornersSB(gray2, (rows, columns), flags)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC + cv.CALIB_USE_EXTRINSIC_GUESS

    ret, CM0, dist0, CM1, dist1, R, T, E, F, rvecs, tvecs, perViewErrors = cv.stereoCalibrateExtended(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), R=R_est, T=T_est, criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv.destroyAllWindows()
    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    #open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)

        k = cv.waitKey(1)
        if k == 27: break

    cv.destroyAllWindows()

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1

def save_extrinsic_calibration_c0_to_c1(R12, T12, prefix = ''):
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    c1_to_c2_rot_trans_filename = os.path.join(prefix, 'cL_to_cR_extrinsiscs.dat') #os.path.join('camera_parameters', prefix + 'cL_to_cR_extrinsiscs.dat')
    outf = open(c1_to_c2_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R12:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T12:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R12, T12

def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join(prefix, 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join(prefix, 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Call with settings filename: "python3 calib.py calibration_settings.yaml calib_num image_resolution"')
        quit()
    
    #Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])
    calib_file_num = sys.argv[2]
    img_resol = sys.argv[3]

    # Directory for data storage
    prefix = f'calibration_parameters\\calibration_data_{img_resol}\\'

    #HOME = os.path.expanduser( '~' ) + '/Projects/LightsCameraGrid/Calibration Data/Stereo_pi/individual_images'
    # HOME = os.path.expanduser( '~' ) + '/Downloads/python_stereo_camera_calibrate/calibration_image_pairs'
    HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1])
    if os.name == 'nt':
        HOME = HOME[:2] + '\\' + HOME[2:]
        print(HOME)

    calib_dir = os.path.join(HOME, f'calibration_code\\calibration_parameters\\calib_{img_resol}')
    if (os.path.isdir(calib_dir) == False):
        os.makedirs(calib_dir)

    CALIBRATION_IMAGE_PAIRS_STEREO = os.path.join(HOME, f'stereopi-setup\\calibration{calib_file_num}\\images_exp{calib_file_num}')#'calibration_image_pairs')
    CALIBRATION_IMAGE_LEFT_CAMREA = os.path.join(HOME, f'stereopi-setup\\calibration{calib_file_num}\\single0')
    CALIBRATION_IMAGE_RIGHT_CAMERA = os.path.join(HOME, f'stereopi-setup\\calibration{calib_file_num}\\single1')
    """Step1. Save calibration frames for single cameras"""

    print('Do you want to recalibrate cameras? (Y/n)')
    user_input = input()
    if user_input.lower() == 'y':
        """Step2. Obtain camera intrinsic matrices and save them"""
        #camera0 intrinsics
        images_prefix = os.path.join(CALIBRATION_IMAGE_PAIRS_STEREO, 'left_camera')
        #print("HERE", images_prefix)
        # calibrate_camera_for_intrinsic_parameters(CALIBRATION_IMAGE_LEFT_CAMREA, f'left_camera_{img_resol}', calib_dir)
        calibrate_camera_for_intrinsic_parameters(images_prefix, f'right_camera_{img_resol}', calib_dir) 
        # save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk

        #HOME = os.path.expanduser( '~' ) + '/Downloads/StereoPiCalibrationImages'
        #camera1 intrinsics
        images_prefix = os.path.join(CALIBRATION_IMAGE_PAIRS_STEREO, 'right_camera')
        # print(images_prefix)
        # calibrate_camera_for_intrinsic_parameters(CALIBRATION_IMAGE_RIGHT_CAMERA, f'right_camera_{img_resol}', calib_dir)
        calibrate_camera_for_intrinsic_parameters(images_prefix, f'right_camera_{img_resol}', calib_dir)
        # save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk

    """Step3. Save calibration frames for both cameras simultaneously"""
    # save_frames_two_cams('camera0', 'camera1') #save simultaneous frames


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    left_camera_parameters = np.load(os.path.join(HOME, 'calibration_code', 'calibration_parameters', f'calib_{img_resol}', f'left_camera_{img_resol}_intrinsics.npz'))
    right_camera_parameters = np.load(os.path.join(HOME, 'calibration_code', 'calibration_parameters', f'calib_{img_resol}', f'right_camera_{img_resol}_intrinsics.npz'))

    cmtx0 = left_camera_parameters['intrinsic']
    dist0 = left_camera_parameters['dist']
    cmtx1 = right_camera_parameters['intrinsic']
    dist1 = right_camera_parameters['dist']

    # cali_path = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1], f'stereopi-setup\\calibration_data\\640_480')
    # if os.name == 'nt':
    #     cali_path = cali_path[:2] + '\\' + cali_path[2:]
    
    # cali_path = cali_path + '\\' + get_calibration_file(cali_path)[0]
    # cmtx0, cmtx1, dist0, dist1, _ = read_mat_calibration(cali_path)


    frames_prefix_c0 = os.path.join(CALIBRATION_IMAGE_PAIRS_STEREO, 'left_camera')
    left_image_names = os.listdir(frames_prefix_c0)
    left_image_paths = [os.path.join(frames_prefix_c0, imname) for imname in left_image_names]

    frames_prefix_c1 = os.path.join(CALIBRATION_IMAGE_PAIRS_STEREO, 'right_camera')
    right_image_names = os.listdir(frames_prefix_c1)
    right_image_paths = [os.path.join(frames_prefix_c1, imname) for imname in right_image_names]

    R_est = np.eye(3)
    T_est = np.array([[-0.24],
                      [-0.004],
                      [-0.01]])
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, left_image_paths, right_image_paths, R_est, T_est)
    
    prefix = f'calibration_parameters\\calibration_data_{img_resol}\\'
    save_extrinsic_calibration_c0_to_c1(R, T, prefix=calib_dir)

    """Step5. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T, prefix=calib_dir) #this will write R and T to disk
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    # check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)


    """Optional. Define a different origin point and save the calibration data"""
    # #get the world to camera0 rotation and translation
    # R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, os.path.join('frames_pair', 'camera0_4.png'))
    # #get rotation and translation from world directly to camera1
    # R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
    #                                           cmtx1, dist1, R1, T1,
    #                                           os.path.join('frames_pair', 'camera0_4.png'),
    #                                           os.path.join('frames_pair', 'camera1_4.png'),)

    # #save rotation and translation parameters to disk
    # save_extrinsic_calibration_parameters(R_W0, T_W0, R_W1, T_W1, prefix = 'world_to_') #this will write R and T to disk

