import gpiozero
import picamera
from picamera import PiCamera
from datetime import datetime
import time


pin = gpiozero.OutputDevice(4)

def main():
   print("Line of code for timing")

   now = datetime.now()
   current_time = now.strftime("%H:%M:%S")
#   print("Current Time Before Pin activation:", current_time)
#   print("unix_timestamp => ",
#      (time.mktime(now.timetuple())))
   t1 = time.time()
   pin.on()
   t2 = time.time()

   tTime = t2 - t1
   #now = datetime.now()
   #current_time = now.strftime("%H:%M:%S")
   #print("unix_timestamp => ",
   #   (time.mktime(now.timetuple())))
   print("Current Time After Pin activation:", tTime)
   pin.off()


   return 0

if __name__ == '__main__':
   main()

