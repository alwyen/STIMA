#!/usr/bin/python
import socket
import cv2
import numpy
import picamera
from picamera import PiCamera
import time
import numpy as np
from datetime import datetime


hostname = socket.gethostname()
print(hostname)
TCP_IP_1 = socket.gethostbyname(hostname)
print(TCP_IP_1)

TCP_IP = '192.168.2.6'
TCP_PORT = 5001

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))


# Camera settimgs
cam_width = 1280 #2560
cam_height = 480 #960

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
#camera.exposure_compensation = 15
#camera.iso = 180
time.sleep(1)

#camera.exposure_mode = 'off'
#camera.exposure_compensation = 25
#camera.brightness = 51

t2 = datetime.now()
img_count = 0
counter = 0
avgtime = 0
# Capture frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True): #, resize=(img_width,img_height)):
    #t1 = datetime.now()
    #frame_show = cv2.resize(frame, (int(img_width*1), int(img_height*1)))
    #cv2.imshow("pair", frame_show)
    key = cv2.waitKey(1) & 0xFF
    if counter == 0:
        counter += 1
        continue
    #t2 = datetime.now()
    # if the `q` key was pressed, break from the loop and save last image
    # if key == ord("p") :
        # # Getting Metadata for Images
        # filename = path + str(img_count) + '.png'
        # cv2.imwrite(filename, frame)
        # img_count += 1
    #encode_param = [int(cv2.IMWRITE_PNG_STRATEGY), 1]
    ret, img_encode = cv2.imencode('.png', frame)
    print(img_encode.shape)
    data = np.array(img_encode)
    stringData = data.tobytes()
    len_stringData = str(len(data)).encode()
    print("TYPE:", type(len_stringData), "DATA:", len_stringData)
    print("TYPE:", type(stringData), "DATA:", len(stringData))
    resulting_str = len_stringData + stringData

    print("TYPE:", type(resulting_str), "DATA:", resulting_str[:20])
#    print(len(stringData))
#    img_decode = cv2.imdecode(data, 1)

#    cv2.imshow('FRAME', img_decode)
#    sock.send(str(len())

    sock.send(resulting_str)

    if key == ord("q"):
        break
    

sock.close()

