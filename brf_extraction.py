import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import shift
import scipy.signal as ss
from scipy.stats.stats import pearsonr
import pandas as pd
from dtw import dtw, accelerated_dtw

from signal_alignment import phase_align, chisqr_align

path_1 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\1'
path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\2'

#make classes man..

ecosmart_CFL_14w = 'ecosmart_CFL_14w'
maxlite_CFL_15w = 'maxlite_CFL_15w'
ge_incandescant_25w = 'ge_incandescant_25w'
philips_incandescent_40w = 'philips_incandescent_40w'
feit_led17p5w = 'feit_led17.5w'

height = 576
width = 1024

def get_rolling_dc_paths(img_path):
    img_paths = path + '\\' + img_path
    rolling = img_paths + '_rolling.jpg'
    dc = img_paths + '_dc.jpg'
    return rolling, dc

def img_from_path(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (width, height))
    return img

def show_plot(array):
    plt.figure(figsize = (10, 2))
    plt.plot(array)
    plt.show()

def show_two_brfs(brf_1, brf_2):
    plt.figure(figsize = (10, 2))
    plt.plot(brf_1, label = '1')
    plt.plot(brf_2, label = '2')
    plt.legend()
    plt.show()

def plot_entire_image(rolling_image):
    height = rolling_image.shape[0]
    width = rolling_image.shape[1]
    for col in range(0, width, 100):
        plt.plot(savitzky_golay_filter(rolling_image[0:height, col], 61, 3))
        plt.show()

def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def brf_extraction(img):
    height = img.shape[0]
    width = img.shape[1]
    column = img[0:height,int(width/2)]
    # column = img[0:height,1239]
    return column

def normalize_img(rolling_img, dc_img):
    normalized_img = np.divide(rolling_img, dc_img)
    height = normalized_img.shape[0]
    width = normalized_img.shape[1]
    normalized_img = normalized_img/(np.sum(normalized_img)/width/height)
    return normalized_img

def normalize_brf(brf):
    brf_points = []
    min = np.amin(brf)
    max = np.amax(brf)
    #FIND A BETTER WAY TO DO THIS (USING NP ARRAY)
    for i in range(len(brf)):
        brf_points.append((brf[i] - min)/(max - min))
    brf_points = np.array(brf_points)
    return brf_points

def process_extract_brf(bulb_path):
    img_rolling, img_dc = get_rolling_dc_paths(bulb_path)    
    img_rolling = img_from_path(img_rolling)
    # img_dc = img_from_path(img_dc)
    brf_rolling = brf_extraction(img_rolling)
    normalized_brf = normalize_brf(brf_rolling)
    return normalized_brf

def moving_average(image_column, window_size):
    average = []
    for x in range(len(image_column)-window_size):
        average.append(np.sum(image_column[x:x+window_size])/window_size/255)
    return average

#maybe just get rid of this..
#second or third order polynomial??
#window_length should be based off of the period length??
def savitzky_golay_filter(brf, window_length, polyorder):
    filtered_data = ss.savgol_filter(brf, window_length, polyorder) #array, window_length, order of polynomial to fit samples; win > poly
    return filtered_data

def crop_brf(brf, start_index, end_index):
    return brf[start_index:end_index]

def align_brfs(brf_1, brf_2):
    #shfit amount corresponds to second argument (brf_2)
    shift_amount = phase_align(brf_1, brf_2, [10, 90]) #[10, 90] => region of interest; figure out what this is???
    shifted_brf_2 = shift(brf_2, shift_amount, mode = 'nearest')
    return brf_1, shifted_brf_2

def pearson_coeff_moving(brf_1, brf_2):
    array = np.array([brf_1, brf_2])
    array = array.reshape(len(brf_1), len(array))
    df = pd.DataFrame(array)

    overall_pearson_r = df.corr(method = 'pearson').iloc[0,1]
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    # out: Pandas computed Pearson r: 0.2058774513561943

    r, p = pearsonr(brf_1, brf_2)
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")

def average_periods(brf):
    # peak_indices = ss.find_peaks(brf)[0]
    peak_indices = ss.find_peaks_cwt(brf, np.arange(1, 70))
    peak_values = brf[peak_indices]
    plt.plot(brf)
    plt.plot(peak_indices, peak_values, 'x')
    plt.show()

def compare_brfs_same_bulb(bulb_path, savgov_window):
    bulb_path_1 = bulb_path + '_0'
    bulb_path_2 = bulb_path + '_1'

    brf_1 = process_extract_brf(bulb_path_1)
    brf_2 = process_extract_brf(bulb_path_2)

    smoothed_brf_1 = savitzky_golay_filter(brf_1, savgov_window, 3)
    smoothed_brf_2 = savitzky_golay_filter(brf_2, savgov_window, 3)
    # smoothed_brf_1 = brf_1
    # smoothed_brf_2 = brf_2

    average_periods(smoothed_brf_1)

    aligned_brf_1, aligned_brf_2 = align_brfs(smoothed_brf_1, smoothed_brf_2)
    show_two_brfs(aligned_brf_1, aligned_brf_2)

    cropped_brf_1 = crop_brf(smoothed_brf_1, 0, 250)
    correlate = ss.correlate(cropped_brf_1, smoothed_brf_2, mode = 'full')/len(cropped_brf_1)
    correlate = crop_brf(correlate, len(cropped_brf_1), len(correlate) - len(cropped_brf_1))
    return correlate

def compare_brfs(bulb_1, bulb_2, savgov_window):
    bulb_1 += '_0'
    bulb_2 += '_1'
    brf_rolling_1 = process_extract_brf(bulb_1)
    brf_rolling_2 = process_extract_brf(bulb_2)

    # brf_rolling_1 = crop_brf(brf_extraction(normalized_1), 0, 500)
    # brf_rolling_2 = crop_brf(brf_extraction(normalized_2), 0, 500)

    smoothed_brf_1 = savitzky_golay_filter(brf_rolling_1, savgov_window, 3)
    smoothed_brf_2 = savitzky_golay_filter(brf_rolling_2, savgov_window, 3)
    # smoothed_brf_1 = brf_rolling_1
    # smoothed_brf_2 = brf_rolling_2

    aligned_brf_1, aligned_brf_2 = align_brfs(smoothed_brf_1, smoothed_brf_2)

    show_two_brfs(aligned_brf_1, aligned_brf_2)

    cropped_brf_1 = crop_brf(smoothed_brf_1, 0, 250)
    correlate_1 = ss.correlate(cropped_brf_1, aligned_brf_2, mode = 'full')/len(cropped_brf_1)
    correlate_1 = crop_brf(correlate_1, len(cropped_brf_1), len(correlate_1) - len(cropped_brf_1))
    return correlate_1

def compare_different_sensitivity_brfs(brf_1, brf_2):
    img_path_1 = path_1 + '\\' + brf_1 + '_rolling.jpg'
    img_path_2 = path + '\\' + brf_2 + '_0_rolling.jpg'
    brf_1 = brf_extraction(img_from_path(img_path_1))
    brf_2 = brf_extraction(img_from_path(img_path_2))
    smoothed_brf_1 = savitzky_golay_filter(brf_1, 61, 3)
    smoothed_brf_2 = savitzky_golay_filter(brf_2, 61, 3)
    normalized_brf_1 = normalize_brf(smoothed_brf_1)
    normalized_brf_2 = normalize_brf(smoothed_brf_2)
    aligned_brf_1, aligned_brf_2 = align_brfs(normalized_brf_1, normalized_brf_2)
    show_two_brfs(aligned_brf_1, aligned_brf_2)

if __name__ == '__main__':
    #I think you need to go back to filtering - losing information in signal?
    compare_different_sensitivity_brfs(ecosmart_CFL_14w, ecosmart_CFL_14w)
    window_size = 61
    for window_size in range(31, 81, 2):
        print(window_size)
        ecosmart_cfl = compare_brfs_same_bulb(ecosmart_CFL_14w, window_size)
        # maxlite_cfl = compare_brfs_same_bulb(maxlite_CFL_15w, window_size)
        # ge_incandescent = compare_brfs_same_bulb(ge_incandescant_25w, window_size)
        # philips_incandescent = compare_brfs_same_bulb(philips_incandescent_40w, window_size)

    cfl_cfl = compare_brfs(ecosmart_CFL_14w, maxlite_CFL_15w, window_size)
    incandescent_incandescent = compare_brfs(ge_incandescant_25w, philips_incandescent_40w, window_size)
    cfl_incandescent_1 = compare_brfs(ecosmart_CFL_14w, ge_incandescant_25w, window_size)
    cfl_incandescent_2 = compare_brfs(ecosmart_CFL_14w, philips_incandescent_40w, window_size)
    cfl_incandescent_3 = compare_brfs(maxlite_CFL_15w, ge_incandescant_25w, window_size)
    cfl_incandescent_4 = compare_brfs(maxlite_CFL_15w, philips_incandescent_40w, window_size)

    # #plots below
    # plt.title('Cross-Correlation Graphs')
    # plt.plot(maxlite_cfl, label = 'Bulb Compared to Itself at Different Position (Maxlite)')
    # plt.plot(ecosmart_cfl, label = 'Bulb Compared to Itself at Different Position (Ecosmart)')
    # plt.plot(cfl_cfl, label = 'Same Type of Bulb, Different Manufacturer')
    # plt.plot(cfl_incandescent_1, label = 'CFL-Incandescent BRF Comparison 1')
    # plt.plot(cfl_incandescent_2, label = 'CFL-Incandescent BRF Comparison 2')
    # plt.plot(cfl_incandescent_3, label = 'CFL-Incandescent BRF Comparison 3')
    # plt.plot(cfl_incandescent_4, label = 'CFL-Incandescent BRF Comparison 4')
    # plt.legend()
    # plt.show()

    # plt.plot(ge_incandescent, label = 'Bulb Compared to Itself at Different Position (GE)')
    # plt.plot(philips_incandescent, label = 'Bulb Compared to Itself at Different Position (Philips)')
    # plt.plot(incandescent_incandescent, label = 'Same Type of Bulb, Different Manufacturer')
    # plt.plot(cfl_incandescent_1, label = 'CFL-Incandescent BRF Comparison 1')
    # plt.plot(cfl_incandescent_2, label = 'CFL-Incandescent BRF Comparison 2')
    # plt.plot(cfl_incandescent_3, label = 'CFL-Incandescent BRF Comparison 3')
    # plt.plot(cfl_incandescent_4, label = 'CFL-Incandescent BRF Comparison 4')
    # plt.legend()
    # plt.show()