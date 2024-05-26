import cv2
import os

height = 960
width = 2560
#HOME = os.path.expanduser( '~' )
HOME = os.path.expanduser( '~' ) + '/Projects/LightsCameraGrid/stereopi-setup/calibration'
def main():
    
    for i in range(1,90):
        #if i in [56,57]: continue
        #print(HOME)
        #path = '/Projects/LightsCameraGrid/Calibration Data/Stereo_Pi/calibration_images'
        img = cv2.imread(HOME + "/scene_{}x{}_{}.png".format(width, height, i))

        img1 = img[:, :int(img.shape[1]/2), :]
        img2 = img[:, int(img.shape[1]/2):, :]
        if not os.path.exists('calibration_image_pairs/left_camera'):
            os.mkdir('calibration_image_pairs/left_camera')
        if not os.path.exists('calibration_image_pairs/right_camera'):
            os.mkdir('calibration_image_pairs/right_camera')

        cv2.imwrite("calibration_image_pairs/left_camera/pair_Left{}.png".format(i), img1)
        cv2.imwrite("calibration_image_pairs/right_camera/pair_Right{}.png".format(i), img2)


if __name__ == "__main__":
    main()