import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\ecosmart_CFL_14w.jpg' #cfl_1
# img_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\maxlite_CFL_15w.jpg' #cfl_1
img_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\ge_incandescant_25w.jpg' #incandescent_1
# img_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\philips_incandescent_40w.jpg' #incandescent_2
# img_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\paper_1.jpg'
height = 576
width = 1024

def img_from_path(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (width, height))
    return img

def show_plot(array):
    plt.plot(array)
    plt.show()

def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def brf_extraction(img):
    height = img.shape[0]
    width = img.shape[1]
    column = img[0:height,int(width/2)]
    # column = img[0:height,1836]
    return column

def moving_average(image_column, window_size):
    average = []
    for x in range(len(image_column)-window_size):
        average.append(np.sum(image_column[x:x+window_size])/window_size/255)
    return average

def crop_brf(brf, start_index, end_index):
    return brf[start_index:end_index]

if __name__ == '__main__':
    img = img_from_path(img_path)
    brf = brf_extraction(img)
    average = moving_average(brf, 5)
    brf = crop_brf(average, 0, 250)
    # show_plot(average)
    show_plot(brf)