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
import sys
import qwiic_titan_gps

args = sys.argv
dataNum = args[1]

qwiicGPS = qwiic_titan_gps.QwiicTitanGps()

if qwiicGPS.connected is False:
    print("Could not connect to to the SparkFun GPS Unit. Double check thatit's wired correctly.", file=sys.stderr)
    exit(1)

qwiicGPS.begin()

# Connecting Board
i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller
sensor = adafruit_bno055.BNO055_I2C(i2c)
time.sleep(0.5)
HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-2], 'Adafruit_CircuitPython_BNO055', 'examples')
cal_file = open(HOME + '/calibration_data.csv')
all_offsets = []

for line in cal_file:
    offsets = line.split(',')
    offset = []
    for data in offsets:
        offset.append(int(data))
    all_offsets.append(offset)
print(all_offsets)
sensor.offsets_magnetometer = (all_offsets[0][0], all_offsets[0][1], all_offsets[0][2])
sensor.offsets_gyroscope = (all_offsets[1][0], all_offsets[1][1], all_offsets[1][2])
sensor.offsets_accelerometer = (all_offsets[2][0], all_offsets[2][1], all_offsets[2][2])


def quat_to_euler(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    # print (q0, q1, q2, q3)
    R00 = q0 * q0 + q1 * q1 -0.5
    R01 = q1 * q2 - q0 * q3
    R02 = q1 * q3 + q0 * q2
    R10 = q1 * q2 + q0 * q3
    R11 = q0 * q0 + q2 * q2 -0.5
    R12 = q2 * q3 - q0 * q1
    R20 = q1 * q3 - q0 * q2
    R21 = q2 * q3 + q0 * q1
    R22 = q0 * q0 + q3 * q3 -0.5

    # Angle from Rotation matrix
#    yaw   =  np.arctan2(R10 , R00) 
#    pitch =  - np.arcsin(R20) #; =  -  asinf(gx);
#    roll  =  np.arctan2(R21 , R22) #; = atan2f(gy , gz);

    # Angle from Tait-Bryan Angles
    yaw   =  np.arctan2(q1 * q2 + q0 * q3, q0 * q0 + q1 * q1 - 0.5)
    pitch =  -  np.arcsin(2.0 * (q1 * q3 - q0 * q2))
    roll  =  np.arctan2(q0 * q1 + q2 * q3, q0 * q0 + q3 * q3 - 0.5)

    print('Yaw:', yaw, 'Pitch:', roll, 'Roll:', pitch)
    
    return yaw, roll, pitch

# Creating metadata file
metaf_dir = "./static_exp/exp{}/imu_data_{}".format(dataNum, dataNum)
if (os.path.isdir(metaf_dir) == False):
    os.makedirs(metaf_dir)

# Initialize CSV file for metadata
csv_file_name = metaf_dir + "/frame_data.csv"
csv_file = open(csv_file_name, 'w')
writer = csv.writer(csv_file)
writer.writerows([['frame', 'X', 'Y', 'Z', 'yaw', 'pitch', 'roll', 'lat', 'lon']])

# Path for captured image
img_dir = "./static_exp/exp{}/scenes_{}".format(dataNum, dataNum)
if (os.path.isdir(img_dir) == False):
    os.makedirs(img_dir)

path = img_dir + '/stereo_image'
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
    #t1 = datetime.now()
    #timediff = t1-t2
    #avgtime = avgtime + (timediff.total_seconds())
    frame_show = cv2.resize(frame, (int(img_width*0.25), int(img_height*0.25)))
    cv2.putText(frame_show, str(img_count), (50,50), font, 2.0, (0,255,0),4, cv2.LINE_AA)
    cv2.imshow("pair", frame_show)
    key = cv2.waitKey(1) & 0xFF
    #t2 = datetime.now()
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("p") :
        #avgtime = avgtime/counter
        #print ("Average time between frames: " + str(avgtime))
        #print ("Average FPS: " + str(1/avgtime))
        print ("Frame Number: " + str(img_count))

        # Getting Metadata for Images
        euler_data = [str(d) for d in sensor.euler]
        yaw, pitch, roll = quat_to_euler(sensor.quaternion)
        q = [yaw, pitch, roll]
        quaternion_data = [str(d) for d in q] #sensor.quaternion]
        gps_data = []
        try:
           if qwiicGPS.get_nmea_data() is True:
               gpsData = [qwiicGPS.gnss_messages['Latitude'], qwiicGPS.gnss_messages['Longitude']]
               gps_data = [str(d) for d in gpsData]
        except: gps_data = [0.0, 0.0]

        line_csv = [['frame' + str(img_count)] + euler_data + quaternion_data + gps_data]
        writer.writerows(line_csv)

        filename = path + str(img_count) + '.png'
        cv2.imwrite(filename, frame)
        img_count += 1
    if key == ord("q"):
        break
    
