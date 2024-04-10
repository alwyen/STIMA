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
   print("Current Time Before Pin activation:", current_time)
   print("unix_timestamp => ",
      (time.mktime(now.timetuple())))
   pin.on()
   now = datetime.now()
   current_time = now.strftime("%H:%M:%S")
   print("unix_timestamp => ",
      (time.mktime(now.timetuple())))
   print("Current Time After Pin activation:", current_time)
   pin.off()


   return 0

if __name__ == '__main__':
   main()

