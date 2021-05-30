import os
import numpy as np
import cv2
import glob
import time

image_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\Rectification\calibration_1'
current_dir = os.getcwd()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

os.chdir(image_path)
images = glob.glob('*.jpg')
num_images = len(images)
image_counter = 1
num_pics_with_corners = 0

for fname in images:
    print(f'Working on {image_counter} / {num_images}')
    
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # print("findChessboardCorners benchmark:")
    # tic = time.time()
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    # toc = time.time() - tic
    # print(f'findChessboardCorners took {toc} seconds')

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(fname)
        num_pics_with_corners += 1
        # print(f'Number of Pictures with Corners Found: {num_pics_with_corners}')
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imwrite('calibration_results/corners_' + fname, img)
    image_counter += 1

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

image_path_2 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\Rectification\Rectification_Test'
os.chdir(image_path_2)

img = cv2.imread('left_0.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print(mtx)
print(dist)
print(newcameramtx)

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)