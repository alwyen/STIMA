import csv
import time
import numpy as np
import os
from datetime import datetime
import board
import adafruit_bno055
import sys

def read_imu(dataNum, calibrate=False):

    # Connecting Board
    i2c = board.I2C()  # uses board.SCL and board.SDA
    # i2c = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    time.sleep(0.5)
    HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-3], 'Adafruit_CircuitPython_BNO055', 'examples')
    cal_file = open(HOME + '/calibration_data.csv')
    all_offsets = []

    if calibrate:
        for line in cal_file:
            offsets = line.split(',')
            offset = []
            for data in offsets:
                offset.append(int(data))
            all_offsets.append(offset)
        print (all_offsets)
        sensor.offsets_magnetometer = (all_offsets[0][0], all_offsets[0][1], all_offsets[0][2])
        sensor.offsets_gyroscope = (all_offsets[1][0], all_offsets[1][1], all_offsets[1][2])
        sensor.offsets_accelerometer = (all_offsets[2][0], all_offsets[2][1], all_offsets[2][2])


    def quat_to_euler(q):
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        # Angle from Tait-Bryan Angles
        yaw   =  np.arctan2(q1 * q2 + q0 * q3, q0 * q0 + q1 * q1 - 0.5)
        pitch =  -  np.arcsin(2.0 * (q1 * q3 - q0 * q2))
        roll  =  np.arctan2(q0 * q1 + q2 * q3, q0 * q0 + q3 * q3 - 0.5)

        return yaw, roll, pitch

    # Creating metadata file
    metaf_dir = "../motion_exp/exp{}/imu_data_{}".format(dataNum, dataNum)
    if (os.path.isdir(metaf_dir) == False):
        os.makedirs(metaf_dir)

    # Initialize CSV file for metadata
    csv_file_name = metaf_dir + "/imu_data.csv"
    csv_file = open(csv_file_name, 'w')
    writer = csv.writer(csv_file)
    writer.writerows([['timestamp', 'yaw', 'pitch', 'roll', 'q0', 'q1', 'q2', 'q3']])

    t2 = datetime.now()

    while( True ):

        quaternion = sensor.quaternion
        date_time = datetime.now()
        timestamp = datetime.timestamp(date_time)*1000
        # Getting Metadata for Images
        yaw, pitch, roll = quat_to_euler(quaternion)
        q = [yaw, pitch, roll]
        euler_data = [str(d) for d in q]
        quaternion_data = [str(d) for d in quaternion]

        line_csv = [[timestamp] + euler_data + quaternion_data]
        writer.writerows(line_csv)

        print (quaternion_data)

if __name__ == '__main__':
    args = sys.argv
    expNum = args[1]
    calibrate = args[2]
    if calibrate.lower() == "t":
        cal = True
    else:
        cal = False

    read_imu(expNum, cal)
