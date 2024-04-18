import picamera
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
from datetime import datetime
import gpiozero
import argparse


parser = argparse.ArgumentParser(description='Process images for specific setting')
parser.add_argument('-c', action='store_true') #, '--calibration')

args = parser.parse_args()

# File for captured image
if args.c:
    filename = './scenes/left/frame'
    dirname = './scenes/left'
else:
    filename = './frames/left/image'
    dirname = './frames/left'

# Camera settimgs
cam_width = 1280
cam_height = 960

# Final image capture settings
scale_ratio = 0.5

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
camera = PiCamera() #(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
camera.hflip = True

pin = gpiozero.InputDevice(4)


t2 = datetime.now()
counter = 0
avgtime = 0
imgNum = 0
# Capture frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    counter+=1
    t1 = datetime.now()
    timediff = t1-t2
    avgtime = avgtime + (timediff.total_seconds())
    cv2.imshow("pair", frame)
    key = cv2.waitKey(1) & 0xFF
    # t2 = datetime.now()
    # if the 'p' key was pressed on main, save the frame
    if (pin.is_active):
        if (os.path.isdir(dirname)==False):
            os.makedirs(dirname)
        #pin.on()
        cv2.imwrite(filename + str(imgNum) + ".png", frame)
        time.sleep(0.200)
        imgNum += 1
        #pin.off()

    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("q") :
        avgtime = avgtime/counter
        print ("Average time between frames: " + str(avgtime))
        print ("Average FPS: " + str(1/avgtime))
        break
