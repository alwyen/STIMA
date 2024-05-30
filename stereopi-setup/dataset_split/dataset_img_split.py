import cv2
import os
import sys
import re

height = 960
width = 2560

def get_max_num_images(home_path):
    img_list = os.listdir(home_path)

    return len(img_list)


#HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1], 'stereopi-setup', 'calibration')
def main():
    if len(sys.argv) < 2:
        print( 'ERROR - Must include experiment number to parse')
        exit(1)
    num = sys.argv[1]
    HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1], 'frames_data')

    #Windows Only
    if os.name == 'nt':
        HOME = HOME[:2] + '\\' + HOME[2:]
    num_images = get_max_num_images(HOME)
    for i in range(0,num_images):
        if i % 10 == 0:
            print(f'Image {i}/{num_images} done.')
            
        img = cv2.imread(HOME + "/frame{}.png".format(i))

        img1 = img[:, :int(img.shape[1]/2), :]
        img2 = img[:, int(img.shape[1]/2):, :]

        if not os.path.exists('dataset_exp{}'.format(num)):
            os.mkdir('dataset_exp{}'.format(num))
        PATH = 'dataset_exp{}'.format(num)
        if not os.path.exists(PATH + '/left_camera'):
            os.mkdir(PATH + '/left_camera')
        if not os.path.exists(PATH + '/right_camera'):
            os.mkdir(PATH + '/right_camera')

        cv2.imwrite(PATH + "/left_camera/frame{}.png".format(i), img1)
        cv2.imwrite(PATH + "/right_camera/frame{}.png".format(i), img2)


if __name__ == "__main__":
    main()
