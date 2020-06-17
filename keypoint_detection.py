import cv2
import numpy as np
import code
import os
#import argparse

def detect_keypoints(path1, save_path):

    # width = 500
    # height = 420

    width = 640
    height = 480

    # width = 2560
    # height = 1920
    
    img1 = cv2.resize(cv2.imread(path1, 0), (width, height))

    # img1 = cv2.imread(path1, 0)

    #SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    sift_kp1, sift_des1 = sift.detectAndCompute(img1, None)

    #SURF detector
    surf = cv2.xfeatures2d.SURF_create()
    surf_kp1, surf_des1 = surf.detectAndCompute(img1, None)

    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    # print(des2)


    # Keypoints (SIFT)
    sift_keypoints = cv2.drawKeypoints(img1, sift_kp1, color=(0, 255, 0), flags=0, outImage = None)

    # Keypoints (SURF)
    surf_keypoints = cv2.drawKeypoints(img1, surf_kp1, color=(0, 255, 0), flags=0, outImage = None)

    # Keypoints (ORB)
    orb_keypoints = cv2.drawKeypoints(img1, kp1, color=(0, 255, 0), flags=0, outImage = None)

    # Image saving paths
    save_img1 = save_path_1 + '/' + 'img1.jpg'
    save_sift = save_path_1 + '/' + 'SIFT_keypoints.jpg'
    save_surf = save_path_1 + '/' + 'SURF_keypoints.jpg'
    save_orb = save_path_1 + '/' + 'ORB_keypoints.jpg'

    #Saving images
    cv2.imwrite(save_img1, img1) #left image
    cv2.imwrite(save_sift, sift_keypoints)
    cv2.imwrite(save_surf, surf_keypoints)
    cv2.imwrite(save_orb, orb_keypoints)

    #Showing images
    cv2.imshow("Img1", img1)
    cv2.imshow("SIFT", sift_keypoints)
    cv2.imshow("SURF", surf_keypoints)
    cv2.imshow("ORB", orb_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_note(message):
    f = open("NOTE.txt", "w")
    f.write(message)
    f.close()

if __name__ == '__main__':
    # message = "lgrt_2_0 (img1) compared with lgrt_2_1 (img2); images practically identical\n\nimages resized to 640x480"
    path1 = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\images\Outdoor_Images\keb_4.jpg'

    # CHECK THIS PATH
    save_path_1 = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\images\KEB_keypoint_test'
    
    os.chdir(save_path_1)
    detect_keypoints(path1, save_path_1)
