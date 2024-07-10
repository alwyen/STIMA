import os
import time
from datetime import datetime
import picamera
from picamera import PiCamera
import cv2
import numpy as np

# Photo session settings
total_photos = 89             # Number of images to take
countdown = 8                 # Interval for count-down timer, seconds
font=cv2.FONT_HERSHEY_SIMPLEX # Cowntdown timer font
 
# Camera settimgs
cam_width = 2560              # Cam sensor width settings
cam_height = 960              # Cam sensor height settings

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
camera.hflip = True


# Initialize Camera Calibration Folder
path = './calibration/scene_'+str(img_width)+'x'+str(img_height)+'_'

# Lets start taking photos! 
counter = 0
t2 = datetime.now()
print ("Starting photo sequence")
for frame in camera.capture_continuous(capture, format="bgra", \
                  use_video_port=True, resize=(cam_width,cam_height)):
    frame_show = cv2.resize(frame, (int(img_width*0.5),int(img_height*0.5)))
    t1 = datetime.now()
    cntdwn_timer = countdown - int ((t1-t2).total_seconds())
    # If cowntdown is zero - let's record next image
    #if cntdwn_timer == -1:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):
      counter += 1
      if (os.path.isdir("./calibration")==False):
            os.makedirs("./calibration")
      filename = path + str(counter) + '.png'
      cv2.imwrite(filename, frame)
      print (' ['+str(counter)+' of '+str(total_photos)+'] '+filename)
      t2 = datetime.now()
      time.sleep(1)
      cntdwn_timer = 0      # To avoid "-1" timer display 
      next
    # Draw cowntdown counter, seconds
    #cv2.putText(frame_show, str(cntdwn_timer), (50,50), font, 2.0, (0,0,255),4, cv2.LINE_AA)
    cv2.imshow("pair", frame_show)
    
    # Press 'Q' key to quit, or wait till all photos are taken
    if (key == ord("q")): #| (counter == total_photos):
      break

 
print ("Photo sequence finished")
