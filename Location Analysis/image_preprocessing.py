import glob
import os
from PIL import Image, ImageOps

def rename_images(folder_path, base_name):
    os.chdir(folder_path)
    image_list = glob.glob('*.jpg')
    for i in range(len(image_list)):
            os.rename(image_list[i], base_name + str(i) + '.jpg')

def rotate_image(folder_path):
    os.chdir(folder_path)
    image_list = glob.glob('*.jpg')
    for img_name in image_list:
        img = Image.open(img_name)
        rot_img = img.rotate(180)
        rot_img.save(img_name)
        print(img_name + ' rotated')

def grayscale(load_path, save_path):
    os.chdir(load_path)
    image_list = glob.glob('*.jpg')
    for image_name in image_list:
        str_name = image_name.split('.')[0]
        img = Image.open(image_name)
        grayscaled = ImageOps.grayscale(img)
        grayscaled.save(save_path + '\\jpg\\' + str_name + '.jpg')
        grayscaled.save(save_path + '\\tif\\' + str_name + '.tif')
        print(str_name + ' done')


if __name__ == '__main__':
    left_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_7_18_21\Left'
    right_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_7_18_21\Right'
    rotate_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_7_18_21\Right\rotate'
    left_save_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_7_18_21\Left_calib_gray'
    right_save_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_7_18_21\Right_calib_gray'
    rename_images(left_path, 'left')
    rename_images(right_path, 'right')

    # might need to run this twice...
    # rotate_image(rotate_path)

    grayscale(left_path, left_save_path)
    grayscale(right_path, right_save_path)