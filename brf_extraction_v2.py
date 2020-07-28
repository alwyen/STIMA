import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import shift
import scipy.signal as ss
from scipy.stats.stats import pearsonr
import pandas as pd

path_1 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\1'
path_2 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\2'
path_3 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\3'
path_4 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1'
# ecosmart_blurred = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\ecosmart_blurred.jpg'
# philips_uncalibrated = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\philips_uncalibrated.jpg'
# philips_calibrated = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\philips_calibrated.jpg'
ecosmart_CFL = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\ecosmart_CFL_14w_0_rolling.jpg'
philips_incandescent = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\philips_incandescent_40w_1_rolling.jpg'
sylvania_CFL = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\sylvania_CFL_13w_1_rolling.jpg'

ecosmart_CFL_14w = 'ecosmart_CFL_14w'
maxlite_CFL_15w = 'maxlite_CFL_15w'
sylvania_CFL_13w = 'sylvania_CFL_13w'
ge_incandescant_25w = 'ge_incandescant_25w'
ge_incandescent_60w = 'ge_incandescent_60w'
philips_incandescent_40w = 'philips_incandescent_40w'
feit_led17p5w = 'feit_led17.5w'

master_brf_list = np.array([ecosmart_CFL_14w, maxlite_CFL_15w, sylvania_CFL_13w, ge_incandescant_25w, philips_incandescent_40w])

height = 576
width = 1024

def get_rolling_dc_paths(img_path):
    rolling = img_path + '_rolling.jpg'
    dc = img_path + '_dc.jpg'
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
        plt.plot(ss.savgol_filter(rolling_image[0:height, col], 61, 3))
        plt.show()

def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def brf_extraction(img):
    height = img.shape[0]
    width = img.shape[1]
    # column = img[0:height,int(width/2)]
    # column = img[771:1272,1047] #ecosmart_blurred
    # column = img[768:1270,1059] #philips_uncalibrated
    # column = img[768:1311,1089] #philips_calibrated
    # column = img[0:height,550] #blurred_1 tests
    column = img[770:1200,550] #blurred_1 tests
    return column

def normalize_brf(brf):
    brf_points = []
    min = np.amin(brf)
    max = np.amax(brf)
    #FIND A BETTER WAY TO DO THIS (USING NP ARRAY)
    for i in range(len(brf)):
        brf_points.append((brf[i] - min)/(max - min))
    brf_points = np.array(brf_points)
    return brf_points

def map(brf, in_min, in_max, out_min, out_max):
    new_brf = []
    for val in brf:
        mapped = (val - in_min)*(out_max - out_min)/(in_max - in_min) + out_min
        new_brf.append(mapped)
    return new_brf

def show_peaks_with_brf(brf):
    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    peak_values = brf[peak_indices]
    plt.plot(peak_indices, peak_values, 'x')
    plt.plot(brf)
    plt.show()

def cycles_from_brf(brf, option): #gets cycles from ONE BRF
    cycle_list = []
    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    for i in range(len(peak_indices)-1):
        cycle = brf[peak_indices[i]:peak_indices[i+1]]
        if option == 'normalize': cycle = normalize_brf(cycle)
        cycle_list.append(cycle)
    return cycle_list

def normalize_half_cycles(brf):
    half_cycles = []
    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    nadir_indices = ss.find_peaks(-brf, distance = 60)[0]
    extrema_indices = np.concatenate((np.array(peak_indices), np.array(nadir_indices)), 0)
    extrema_indices = sorted(extrema_indices)
    extrema_values = brf[extrema_indices]
    for i in range(len(extrema_indices)-1):
        half_cycle = brf[extrema_indices[i]:extrema_indices[i+1]]
        half_cycle = normalize_brf(half_cycle)
        half_cycles.append(half_cycle)
    return half_cycles

#first figure out how many cycles 
def remove_sin(norm_brf):
    fitted_sinusoid = np.array([]) #concatenate sinusoids to fit brf
    norm_brf = np.array(norm_brf)
    
    min = np.amin(norm_brf)
    max = np.amax(norm_brf)
    
    value_first = int(round(norm_brf[0]))
    value_last = int(round(norm_brf[len(norm_brf)-1]))
    
    brf = map(norm_brf, min, max, -1, 1)
    brf = np.array(brf)
    brf_temp = np.copy(brf)

    start = np.pi/2
    stop = -np.pi/2*3

    if value_first == 0: #initial values need to be flipped if starting at 0
        brf_temp = -brf_temp
        start = -start
        stop = -stop

    indices = ss.find_peaks(brf_temp, distance = 60)[0]
    x = np.linspace(start, stop, indices[0]+1) #peak_indices[0] + 1 b/c brf starts at 0?
    fitted_sinusoid = np.sin(x)
    for i in range(len(indices)-1):
        x = np.linspace(start, stop, indices[i+1]-indices[i])
        fitted_sinusoid = np.concatenate((fitted_sinusoid, np.sin(x)), 0)
    if value_first != value_last:
        x = np.linspace(start, stop/3, (len(brf)-1) - indices[len(indices)-1])
        fitted_sinusoid = np.concatenate((fitted_sinusoid, np.sin(x)), 0)

    if len(brf) != len(fitted_sinusoid): #TEMPORARY FIX
        fitted_sinusoid = np.concatenate((fitted_sinusoid, [value_last]), 0)
    
    new_brf = np.subtract(brf, fitted_sinusoid)
    # plt.plot(brf, label = 'original')
    # plt.plot(fitted_sinusoid, label = 'fitted')
    plt.plot(new_brf, label = 'subtracted')
    
    # x = np.arrange
    # plt.plot(np.sin())
    plt.show()
    print(f'First: {value_first}\nLast: {value_last}')

def extract_normalized_brf(brf):
    new_brf = np.array([])
    cycle_list = normalize_half_cycles(brf)
    for cycle in cycle_list:
        new_brf = np.concatenate((new_brf, cycle), 0)
    remove_sin(new_brf)
    show_plot(new_brf)

def pearson_coeff(brf_1, brf_2):
    if len(brf_1) > len(brf_2): brf_1 = brf_1[0:len(brf_2)]
    elif len(brf_2) > len(brf_1): brf_2 = brf_2[0:len(brf_1)]
    r, p = pearsonr(brf_1, brf_2)
    return r

def cross_corr(brf_1, brf_2):
    correlate = ss.correlate(brf_1, brf_2, mode = 'full')/len(brf_1)
    max = round(np.amax(np.array(correlate)), 3)
    return max

def extract_brfs_from_list(master_brf_list):
    brf_list_1 = []
    brf_list_2 = []

    for bulb in master_brf_list:
        # brf_path_1 = path_3 + '\\' + bulb + '_0_rolling.jpg'
        # brf_path_2 = path_3 + '\\' + bulb + '_1_rolling.jpg'
        brf_path_1 = path_4 + '\\' + bulb + '_0_rolling.jpg'
        brf_path_2 = path_4 + '\\' + bulb + '_1_rolling.jpg'

        img_1 = img_from_path(brf_path_1)
        img_2 = img_from_path(brf_path_2)

        # brf_1 = normalize_brf(brf_extraction(img_1))
        # brf_2 = normalize_brf(brf_extraction(img_2))
        brf_1 = img_1[740:1230,550]
        brf_2 = img_2[590:1050,2220]

        brf_list_1.append(brf_1)
        brf_list_2.append(brf_2)
    
    return brf_list_1, brf_list_2

def normalize_brfs_from_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []

    for i in range(len(brf_list_1)):
        list_1.append(normalize_brf(brf_list_1[i])) #window, polynomial order
        list_2.append(normalize_brf(brf_list_2[i]))

    return list_1, list_2

def smooth_brfs_from_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []

    for i in range(len(brf_list_1)):
        list_1.append(ss.savgol_filter(brf_list_1[i], 51, 3)) #window, polynomial order
        list_2.append(ss.savgol_filter(brf_list_2[i], 51, 3))

    return list_1, list_2

def normalize_smoothed_brf_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []
    normalized_1 = np.array([])
    normalized_2 = np.array([])
    for i in range(len(brf_list_1)):
        cycle_list_1 = cycles_from_brf(brf_list_1[i], 'normalize')
        cycle_list_2 = cycles_from_brf(brf_list_2[i], 'normalize')
        #reconstruct each BRF
        if len(cycle_list_1) == len(cycle_list_2):
            for j in range(len(cycle_list_1)): #same length arrays so okay
                normalized_1 = np.concatenate((normalized_1,cycle_list_1[j]), 0)
                normalized_2 = np.concatenate((normalized_2,cycle_list_2[j]), 0)
        else:
            for cycle in cycle_list_1:
                normalized_1 = np.concatenate((normalized_1, cycle), 0)
            for cycle in cycle_list_2:
                normalized_2 = np.concatenate((normalized_2, cycle), 0)
        list_1.append(normalized_1) #appends new normalized BRFs
        list_2.append(normalized_2)
        normalized_1 = np.array([]) #reset arrays
        normalized_2 = np.array([])
    return list_1, list_2

def normalize_brf_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []
    for i in range(len(brf_list_1)):
        list_1.append(normalize_brf(brf_list_1[i]))
        list_2.append(normalize_brf(brf_list_2[i]))
    return list_1, list_2

#name for list 2 will be reverse of name for list 1
def correlation_heat_map(brf_list_1, brf_list_2, title):
    peak_cross_corr_list = []
    cross_corr_heatmap = []

    brf_name_list_1 = np.array(master_brf_list)
    brf_name_list_2 = np.flip(brf_name_list_1)
    brf_list_2 = np.flip(np.array(brf_list_2))

    #double checked with x/y-axis is correct; seems correct
    for brf_1 in brf_list_1:
        for brf_2 in brf_list_2:
            # plt.plot(brf_1)
            # plt.plot(brf_2)
            # plt.show()
            max = cross_corr(brf_1, brf_2)
            peak_cross_corr_list.append(max)
        cross_corr_heatmap.append(peak_cross_corr_list)
        peak_cross_corr_list = []
    
    fig, ax = plt.subplots()
    # pcoeff_heatplot = ax.imshow(pcoeff_heatmap, cmap = "Blues")
    cross_corr_heatplot = ax.imshow(cross_corr_heatmap, cmap = "Blues")
    ax.set_xticks(np.arange(len(brf_list_1)))
    ax.set_yticks(np.arange(len(brf_list_2)))
    ax.set_xticklabels(brf_name_list_2)

    for label in ax.xaxis.get_ticklabels():
        label.set_horizontalalignment('right')

    ax.set_yticklabels(brf_name_list_1)
    ax.set_title(title)

    for i in range(len(cross_corr_heatmap)):
        for j in range(len(cross_corr_heatmap[i])):
            text = ax.text(j, i, cross_corr_heatmap[i][j],
                        ha="center", va="center", color="black")

    colorbar = fig.colorbar(cross_corr_heatplot)
    plt.xticks(rotation = 45)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    img_1 = img_from_path(ecosmart_CFL)
    img_2 = img_from_path(philips_incandescent)
    img_3 = img_from_path(sylvania_CFL)

    brf_1 = brf_extraction(img_1)
    # brf_2 = brf_extraction(img_2)
    # brf_3 = brf_extraction(img_3)
    brf_2 = img_2[590:1050,2220]
    brf_3 = img_3[590:1050,2220]

    smoothed_1 = ss.savgol_filter(brf_1, 51, 3)
    smoothed_2 = ss.savgol_filter(brf_2, 51, 3)
    smoothed_3 = ss.savgol_filter(brf_3, 51, 3)

    normalized_1 = normalize_brf(smoothed_1)
    normalized_2 = normalize_brf(smoothed_2)
    normalized_3 = normalize_brf(smoothed_3)

    extract_normalized_brf(normalized_1)
    extract_normalized_brf(normalized_2)
    extract_normalized_brf(normalized_3)

    max_1 = cross_corr(normalized_1, normalized_2)
    # max_1 = cross_corr(normalized_2, normalized_1)
    # max_2 = cross_corr(normalized_2, normalized_3)
    max_3 = cross_corr(normalized_1, normalized_3)
    # max_3 = cross_corr(normalized_3, normalized_1)

    print(max_1)
    # print(max_2)
    print(max_3)

    # brf_list_1, brf_list_2 = extract_brfs_from_list(master_brf_list)
    # smoothed_list_1, smoothed_list_2 = smooth_brfs_from_list(brf_list_1, brf_list_2)
    # norm_smooth_list_1, norm_smooth_list_2 = normalize_brfs_from_list(smoothed_list_1, smoothed_list_2)

    # # norm_smooth_list_1, norm_smooth_list_2 = normalize_smoothed_brf_list(smoothed_list_1, smoothed_list_2)
    # # cycle_list_1, cycle_list_2 = extract_cycles_from_list(smoothed_list_1, smoothed_list_2)
    # # cycle_list_1, cycle_list_2 = extract_cycles_from_list(norm_smooth_list_1, norm_smooth_list_2)
    # # normalized_cycle_list_1, normalized_cycle_list_2 = normalize_brf_list(cycle_list_1, cycle_list_2)

    # correlation_heat_map(brf_list_1, brf_list_2, 'Raw BRF Cross Correlation Heat Map')
    # # correlation_heat_map(smoothed_list_1, smoothed_list_2, 'Smoothed BRF Cross Correlation Heat Map')
    # correlation_heat_map(norm_smooth_list_1, norm_smooth_list_2, 'Normalized-Smoothed BRF Cross Correlation Heat Map')
    # # correlation_heat_map(smoothed_list_1, smoothed_list_2, 'Cycle Cross Corrlation Heat Map') #??
    # correlation_heat_map(cycle_list_1, cycle_list_2, 'Averaged Cycle Cross Correlation Heat Map')
    # correlation_heat_map(normalized_cycle_list_1, normalized_cycle_list_2, 'Normalized-Averaged Cycle Cross Correlation Heat Map')