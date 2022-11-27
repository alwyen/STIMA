import glob
import os
from PIL import Image, ImageOps

'''
converts all image names from phones into [base_name]-numbered image names
manually put in folder paths
'''
def rename_images(folder_path, base_name):
    os.chdir(folder_path)
    image_list = glob.glob('*.jpg')
    image_list.sort()
    print(base_name)
    for i in range(len(image_list)):
            print(image_list[i])
            os.rename(image_list[i], base_name + str(i) + '.jpg')
    print()
'''
sometimes, images come out rotated - put rotated images into separate folder
'''
def rotate_image(folder_path):
    os.chdir(folder_path)
    image_list = glob.glob('*.jpg')
    for img_name in image_list:
        img = Image.open(img_name)
        rot_img = img.rotate(180)
        rot_img.save(img_name)
        print(img_name + ' rotated')

'''
convert color images to grayscale images for matlab camera calibration
'''
def grayscale(load_path, save_path):
    os.chdir(load_path)
    image_list = glob.glob('*.jpg')
    for image_name in image_list:
        str_name = image_name.split('.')[0]
        img = Image.open(image_name)
        grayscaled = ImageOps.grayscale(img)
        # grayscaled.save(save_path + '\\jpg\\' + str_name + '.jpg')
        # grayscaled.save(save_path + '\\tif\\' + str_name + '.tif')
        grayscaled.save(save_path + os.sep + str_name + '.tif')
        print(str_name + ' done')


if __name__ == '__main__':
    # left_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_8_4_21\Left'
    # right_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_8_4_21\Right'

    # left_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_8_11_21\Left_Near'
    # right_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_8_11_21\Right_Near'

    # left_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\images_10_08_21\Left'
    # right_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\images_10_08_21\Right'

    # rotate_path = r'C:\Users\alexy\Dropbox\UCSD\Research\Stereo\Camera Calibration\calibration_8_4_21\Right\rotate'

    left_path = r'/Users/alexyen/Dropbox/UCSD/Research/Stereo/Camera Calibration/calibration_7_11_22/Left'
    right_path = r'/Users/alexyen/Dropbox/UCSD/Research/Stereo/Camera Calibration/calibration_7_11_22/Right'

    left_save_path = r'/Users/alexyen/Dropbox/UCSD/Research/Stereo/Camera Calibration/calibration_7_11_22/Left_Gray'
    right_save_path = r'/Users/alexyen/Dropbox/UCSD/Research/Stereo/Camera Calibration/calibration_7_11_22/Right_Gray'

    # rename_images(left_path, 'left')
    # rename_images(right_path, 'right')

    # might need to run this twice...
    # rotate_image(rotate_path)

    grayscale(left_path, left_save_path)
    grayscale(right_path, right_save_path)