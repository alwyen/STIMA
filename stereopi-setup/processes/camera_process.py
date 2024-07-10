import csv
import picamera
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
from datetime import datetime
import sys

def frame_capture(dataNum):
    # Path for captured image
    img_dir = "../motion_exp/exp{}/scenes_{}".format(dataNum, dataNum)
    if (os.path.isdir(img_dir) == False):
        os.makedirs(img_dir)

    path = img_dir + '/stereo_image'
    font=cv2.FONT_HERSHEY_SIMPLEX # Fame Num. font

    # Creating metadata file
    metaf_dir = "../motion_exp/exp{}/camera_data{}".format(dataNum, dataNum)
    if (os.path.isdir(metaf_dir) == False):
        os.makedirs(metaf_dir)

    # Initialize CSV file for metadata
    csv_file_name = metaf_dir + "/frame_data.csv"
    csv_file = open(csv_file_name, 'w')
    writer = csv.writer(csv_file)
    writer.writerows([['frame', 'timestamp']])

    # Camera settimgs
    cam_width = 2560
    cam_height = 960

    # Final image capture settings
    scale_ratio = 1

    # Camera resolution height must be dividable by 16, and width by 32
    cam_width = int((cam_width+31)/32)*32
    cam_height = int((cam_height+15)/16)*16
    print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

    # Buffer for captured image settings
    img_width = int (cam_width * scale_ratio)
    img_height = int (cam_height * scale_ratio)
    capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

    # Initialize the camera
    camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
    camera.resolution=(cam_width, cam_height)
    camera.framerate = 20
    camera.hflip = True

    img_count = 0
    # Capture frames from the camera
    for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True): #, resize=(img_width,img_height)):
        date_time = datetime.now()
        timestamp = datetime.timestamp(date_time)*1000

        frame_show = cv2.resize(frame, (int(img_width*0.25), int(img_height*0.25)))
        cv2.putText(frame_show, str(img_count), (50,50), font, 2.0, (0,255,0),4, cv2.LINE_AA)
        cv2.imshow("pair", frame_show)

        filename = path + str(img_count) + '.png'
        cv2.imwrite(filename, frame)
        img_count += 1

        # Getting Metadata for images
        line_csv = [['frame' + str(img_count)] + [timestamp]]
        writer.writerows(line_csv)

        key = cv2.waitKey(1) & 0xFF
        # if the `f` key was pressed, break from the loop and finish process
        if key == ord("f"):
            break

if __name__ == '__main__':
    args = sys.argv
    expNum = args[1]

    frame_capture(expNum)
