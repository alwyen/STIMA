import gpiozero
import picamera
from picamera import PiCamera
from datetime import datetime
import time


pin = gpiozero.InputDevice(4)

def main():
   print("Line of code for timing")

   now = datetime.now()
   current_time = now.strftime("%H:%M:%S")
#   print("Current Time Before Pin activation:", current_time)
#   print("unix_timestamp => ",
#      (time.mktime(now.timetuple())))
   time1 = time.time()
   check = pin.is_active
   while(not check): 
       time1 = time.time()
       check = pin.is_active
       #time1 = time.time()
   time2 = time.time()
   #pin.on()
   timePassed = time2 - time1
   now = datetime.now()
#   current_time = now.strftime("%H:%M:%S")
#   print("unix_timestamp => ",
#      (time.mktime(now.timetuple())))
   print("Current Time Passed", timePassed)
   #pin.off()


   return 0

if __name__ == '__main__':
   main()

