# Copyright (C) 2019 Eugene Pomazov, <stereopi.com>, virt2real team
#
# This file is part of StereoPi tutorial scripts.
#
# StereoPi tutorial is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# StereoPi tutorial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoPi tutorial.  
# If not, see <http://www.gnu.org/licenses/>.
#
# Most of this code is updated version of 3dberry.org project by virt2real
# 
# Thanks to Adrian and http://pyimagesearch.com, as there are lot of
# code in this tutorial was taken from his lessons.
# 

import csv
import picamera
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
from datetime import datetime
import board
import adafruit_bno055


# Connecting Board
i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller
sensor = adafruit_bno055.BNO055_I2C(i2c)

# Creating metadata file
if (os.path.isdir("./imu_data") == False):
    os.makedirs("./imu_data")

# Initialize CSV file for metadata
csv_file = open("./imu_data/frame_data.csv", 'w')
writer = csv.writer(csv_file)
writer.writerows([['frame', 'X', 'Y', 'Z', 'q1', 'q2', 'q3', 'q4']])

# Path for captured image
path = './scenes/stereo_image'
font=cv2.FONT_HERSHEY_SIMPLEX # Fame Num. font


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


t2 = datetime.now()
img_count = 0
counter = 0
avgtime = 0
# Capture frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True): #, resize=(img_width,img_height)):
    counter+=1
    t1 = datetime.now()
    timediff = t1-t2
    avgtime = avgtime + (timediff.total_seconds())
    frame_show = cv2.resize(frame, (int(img_width*0.25), int(img_height*0.25)))
    cv2.putText(frame_show, str(img_count), (50,50), font, 2.0, (0,255,0),4, cv2.LINE_AA)
    cv2.imshow("pair", frame_show)
    key = cv2.waitKey(1) & 0xFF
    t2 = datetime.now()
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("p") :
        avgtime = avgtime/counter
        print ("Average time between frames: " + str(avgtime))
        print ("Average FPS: " + str(1/avgtime))
        print("Frame Number: " + str(img_count))
        if (os.path.isdir("./scenes") == False):
            os.makedirs("./scenes")
        euler_data = [str(d) for d in sensor.euler]
        quaternion_data = [str(d) for d in sensor.quaternion]
        line_csv = [['frame' + str(img_count)] + euler_data + quaternion_data]
        writer.writerows(line_csv)
        filename = path + str(img_count) + '.png'
        cv2.imwrite(filename, frame)
        img_count += 1
    if key == ord("q"):
        break
    
