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
# Example Function Call: python dataset_img_split.py 
def main():
    if len(sys.argv) < 2:
        print( 'ERROR - Must include experiment Directory/Number to parse: EX - python script.py [Dir] [ExpNum]')
        exit(1)
    directory = sys.argv[1]
    images_save_path = ''
    for i in range(len(sys.argv[1].split('\\')) - 1):
        images_save_path += '/' + sys.argv[1].split('\\')[i]
    num = sys.argv[2]
    
    HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1], directory)
    HOME_IMG_PATH = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1], images_save_path[1:])

    print(HOME)
    print(HOME_IMG_PATH)
    #Windows Only
    if os.name == 'nt':
        HOME = HOME[:2] + '\\' + HOME[2:]
        HOME_IMG_PATH = HOME_IMG_PATH[:2] + '\\' + HOME_IMG_PATH[2:]
        print(HOME)
    num_images = get_max_num_images(HOME)
    for i in range(1,num_images+1):
        if i % 10 == 0:
            print(f'Image {i}/{num_images} done.')
            
        #img = cv2.imread(HOME + "/stereo_image{}.png".format(i))
        img = cv2.imread(HOME + "/scene_2560x960_{}.png".format(i))

        img1 = img[:, :int(img.shape[1]/2), :]
        img2 = img[:, int(img.shape[1]/2):, :]
        
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
        
        if not os.path.exists(HOME_IMG_PATH + '/images_exp{}'.format(num)):
            os.mkdir(HOME_IMG_PATH + '/images_exp{}'.format(num))
        PATH = HOME_IMG_PATH + '/images_exp{}'.format(num)
        if not os.path.exists(PATH + '/left_camera'):
            os.mkdir(PATH + '/left_camera')
        if not os.path.exists(PATH + '/right_camera'):
            os.mkdir(PATH + '/right_camera')

        cv2.imwrite(PATH + "/left_camera/frame{}.png".format(i), img1)
        cv2.imwrite(PATH + "/right_camera/frame{}.png".format(i), img2)


if __name__ == "__main__":
    main()
