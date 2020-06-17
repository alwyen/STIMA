import cv2
import numpy as np

if __name__ == "__main__":
    array = np.zeros((360, 640, 1), dtype = np.uint8)
    cv2.line(array, (0,0), (360, 360), 255, 1)
    cv2.imshow('test', array)
    cv2.waitKey(0)
    cv2.destroyAllWindowns()