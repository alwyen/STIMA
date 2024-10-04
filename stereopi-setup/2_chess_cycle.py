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
import sys
import os
import time
from datetime import datetime
import picamera
from picamera import PiCamera
import cv2
import numpy as np

if len(sys.argv) < 4:
    print("Must use with script call: 'python 2_chess_cycle.py [calib_num] [width] [height]'")
    exit(1)

# Photo session settings
total_photos = 205             # Number of images to take
countdown = 5                 # Interval for count-down timer, seconds
font=cv2.FONT_HERSHEY_SIMPLEX # Cowntdown timer font

calib_num = sys.argv[1] 

# Camera settimgs
cam_width = int(sys.argv[2])               # Cam sensor width settings
cam_height = int(sys.argv[3])              # Cam sensor height settings

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
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
camera.hflip = False


# Initialize Camera Calibration Folder
path = f'./calibration{calib_num}/scene_'+str(img_width)+'x'+str(img_height)+'_'

# Lets start taking photos! 
counter = 0
t2 = datetime.now()
print ("Starting photo sequence")
for frame in camera.capture_continuous(capture, format="bgra", \
                  use_video_port=True, resize=(cam_width,cam_height)):
    
    t1 = datetime.now()
    cntdwn_timer = countdown - int ((t1-t2).total_seconds())
    # If cowntdown is zero - let's record next image
    if cntdwn_timer == -1:
      counter += 1
      if (os.path.isdir(f"./calibration{calib_num}")==False):
            os.makedirs(f"./calibration{calib_num}")
      filename = path + str(counter) + '.png'
      cv2.imwrite(filename, frame)
      print (' ['+str(counter)+' of '+str(total_photos)+'] '+filename)
      t2 = datetime.now()
      time.sleep(1)
      cntdwn_timer = 0      # To avoid "-1" timer display 
      next
    # Draw cowntdown counter, seconds
    if img_width > 1280:
        frame = cv2.resize(frame, (int(img_width*0.5),int(img_height*0.5)))

    cv2.putText(frame, str(cntdwn_timer), (50,90), font, 3.5, (0,0,255),4, cv2.LINE_AA)
    cv2.imshow("pair", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'Q' key to quit, or wait till all photos are taken
    if (key == ord("q")) | (counter == total_photos):
      break

 
print ("Photo sequence finished")
 
