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

import qwiic_titan_gps


qwiicGPS = qwiic_titan_gps.QwiicTitanGps()

if qwiicGPS.connected is False:
    print("Could not connect to to the SparkFun GPS Unit. Double check thatit's wired correctly.", file=sys.stderr)
    exit(1)

qwiicGPS.begin()

# Connecting Board
i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller
sensor = adafruit_bno055.BNO055_I2C(i2c)

# Creating metadata file
if (os.path.isdir("./imu_data2") == False):
    os.makedirs("./imu_data2")

# Initialize CSV file for metadata
csv_file = open("./imu_data2/frame_data.csv", 'w')
writer = csv.writer(csv_file)
writer.writerows([['frame', 'X', 'Y', 'Z', 'q1', 'q2', 'q3', 'q4', 'lat', 'lon']])

# Path for captured image
path = './frames_data/frame'
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
    #frame_show = cv2.resize(frame, (int(img_width*0.25), int(img_height*0.25)))
    #cv2.putText(frame_show, str(img_count), (50,50), font, 2.0, (0,255,0),4, cv2.LINE_AA)
    #cv2.imshow("pair", frame_show)
    key = cv2.waitKey(1) & 0xFF
    t2 = datetime.now()
    # if the `q` key was pressed, break from the loop and save last image

    if counter > 3:
        counter = 0
        #avgtime = avgtime/counter
        #print ("Average time between frames: " + str(avgtime))
        #print ("Average FPS: " + str(1/avgtime))
        print ("Frame Number: " + str(img_count))
        if (os.path.isdir("./frames_data") == False):
            os.makedirs("./frames_data")

        # Getting Metadata for Images
        euler_data = [str(d) for d in sensor.euler]
        quaternion_data = [str(d) for d in sensor.quaternion]
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
