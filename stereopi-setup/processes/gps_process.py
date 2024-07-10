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

'''
def read_gps(dataNum, calibrate=False):

    qwiicGPS = qwiic_titan_gps.QwiicTitanGps()

    if qwiicGPS.connected is False:
        print("Could not connect to to the SparkFun GPS Unit. Double check thatit's wired correctly.", file=sys.stderr)
        exit(1)

    qwiicGPS.begin()


    # Creating metadata file
    metaf_dir = "../motion_exp/exp{}/gps_data_{}".format(dataNum, dataNum)
    if (os.path.isdir(metaf_dir) == False):
        os.makedirs(metaf_dir)

    # Initialize CSV file for metadata
    csv_file_name = metaf_dir + "/gps_data.csv"
    csv_file = open(csv_file_name, 'w')
    writer = csv.writer(csv_file)
    writer.writerows([['timestamp', 'longitude', 'latitude', 'altitude']])

    t2 = datetime.now()

    while( True ):

        gps_data = []
        try:
           if qwiicGPS.get_nmea_data() is True:
               gpsData = [qwiicGPS.gnss_messages['Latitude'], qwiicGPS.gnss_messages['Longitude', qwiicGPS.gnss_messages['Altitude']]]
               gps_data = [str(d) for d in gpsData]
        except: gps_data = [0.0, 0.0, 0.0]

        date_time = datetime.now()
        timestamp = datetime.timestamp(date_time)*1000

        # Getting Metadata for Images

        line_csv = [[timestamp] + gps_data]
        writer.writerows(line_csv)
        time.sleep(0.5)
        print (gps_data)

if __name__ == '__main__':
    args = sys.argv
    expNum = args[1]

    read_gps(expNum)
'''
def run_example(dataNum):
    
    # Creating metadata file
    metaf_dir = "../motion_exp/exp{}/gps_data_{}".format(dataNum, dataNum)
    if (os.path.isdir(metaf_dir) == False):
        os.makedirs(metaf_dir)

    # Initialize CSV file for metadata
    csv_file_name = metaf_dir + "/gps_data.csv"
    csv_file = open(csv_file_name, 'w')
    writer = csv.writer(csv_file)
    writer.writerows([['timestamp', 'longitude', 'latitude', 'altitude']])

    print("SparkFun GPS Breakout - XA1110!")
    qwiicGPS = qwiic_titan_gps.QwiicTitanGps()

    if qwiicGPS.connected is False:
        print("Could not connect to to the SparkFun GPS Unit. Double check that\
              it's wired correctly.", file=sys.stderr)
        return

    qwiicGPS.begin()

    while True:
        try:
           if qwiicGPS.get_nmea_data() is True:
#               print("Latitude: {}, Longitude: {}, Time: {}".format(
#                qwiicGPS.gnss_messages['Latitude'],
#                qwiicGPS.gnss_messages['Longitude'],
#                qwiicGPS.gnss_messages['Altitude']))
               gpsData = [qwiicGPS.gnss_messages['Latitude'], qwiicGPS.gnss_messages['Longitude'], qwiicGPS.gnss_messages['Altitude']]
               gps_data = [str(d) for d in gpsData]

               date_time = datetime.now()
               timestamp = datetime.timestamp(date_time)*1000

               # Getting Metadata for Images

               line_csv = [[timestamp] + gps_data]
               writer.writerows(line_csv)

               print (gps_data) 
        except: continue
        time.sleep(1)

if __name__ == '__main__':
    args = sys.argv
    expNum = args[1]
    try:
        run_example(expNum)
    except (KeyboardInterrupt, SystemExit) as exErr:
        print("Ending Basic Example.")
        sys.exit(0)
