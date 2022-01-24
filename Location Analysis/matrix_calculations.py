import numpy as np

def Homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))

def Dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]

def calc_camera_calibration_matrix(image_width, image_height, pixel_width, pixel_height, focal_length):
    alpha_x = focal_length / pixel_width
    alpha_y = focal_length / pixel_height

    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2

    K = np.zeros((3,3))

    K[0][0] = alpha_x
    K[0][1] = 0
    K[0][2] = x0
    K[1][0] = 0
    K[1][1] = alpha_y
    K[1][2] = y0
    K[2][0] = 0
    K[2][1] = 0
    K[2][2] =  1

    return K

if __name__ == '__main__':
    # pixel_1; focal_length and pixel_width/pixel_height need to be in same units
    pix1_image_width = 3044
    pix1_image_height = 4048
    pix1_sensor_width = 4.72 #in mm
    pix1_sensor_height = 6.27 #in mm
    pix1_pixel_width = pix1_sensor_width / pix1_image_width * 1000      #units: um/pixel
    pix1_pixel_height = pix1_sensor_height / pix1_image_height * 1000   #units: um/pixel
    pix1_focal_length = 4.67 * 1000                                     #units: um/pixel

    # pixel_2
    pix2_image_width = 3032
    pix2_image_height = 4032
    pix2_sensor_width = 4.24 #in mm
    pix2_sensor_height = 5.64 #in mm
    pix2_pixel_width = pix2_sensor_width / pix2_image_width * 1000      #units: um/pixel
    pix2_pixel_height = pix2_sensor_height / pix2_image_height * 1000   #units: um/pixel
    pix2_focal_length = 4.4589 * 1000                                   #units: um/pixel

    # S9
    s9_image_width = 3024
    s9_image_height = 4032
    s9_sensor_width = 4.23 #in mm
    s9_sensor_height = 5.64 #in mm
    s9_pixel_width = s9_sensor_width / s9_image_width * 1000      #units: um/pixel
    s9_pixel_height = s9_sensor_height / s9_image_height * 1000   #units: um/pixel
    s9_focal_length = 4.3 * 1000                                   #units: um/pixel

    K1 = calc_camera_calibration_matrix(pix1_image_width, pix1_image_height, pix1_pixel_width, pix1_pixel_height, pix1_focal_length)
    K2 = calc_camera_calibration_matrix(pix2_image_width, pix2_image_height, pix2_pixel_width, pix2_pixel_height, pix2_focal_length)
    K3 = calc_camera_calibration_matrix(s9_image_width, s9_image_height, s9_pixel_width, s9_pixel_height, s9_focal_length)

    print('Pixel XL')
    print(K1)

    print()

    print('Pixel 2 XL')
    print(K2)

    print()

    print('Samsung Galaxy S9')
    print(K3)