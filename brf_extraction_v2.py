import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import shift
import scipy.signal as ss
from scipy.fftpack import fft
from scipy.stats.stats import pearsonr
import pandas as pd
import os.path
from os import path
import math
import pylab

from signal_alignment import phase_align

path_1 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\1'
path_2 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\2'
path_3 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\3'
path_4 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1'

sylv_file_num = 'sylvania_0'
subtr_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF Subtraction Images 2\sylvania' + '\\' + sylv_file_num + '.jpg'
brf_save_load_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\csv_brf'
philips_file_num = 'philips_4'
# subtr_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF Subtraction Images 2\philips' + '\\' + philips_file_num + '.jpg'

# ecosmart_blurred = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\ecosmart_blurred.jpg'
# philips_uncalibrated = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\philips_uncalibrated.jpg'
# philips_calibrated = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\philips_calibrated.jpg'
ecosmart_CFL = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\ecosmart_CFL_14w_0_rolling.jpg'
philips_incandescent = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\philips_incandescent_40w_1_rolling.jpg'
sylvania_CFL = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\sylvania_CFL_13w_1_rolling.jpg'
ge_incandescant = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\ge_incandescant_25w_0_rolling.jpg'

ecosmart_CFL_14w = 'ecosmart_CFL_14w'
maxlite_CFL_15w = 'maxlite_CFL_15w'
sylvania_CFL_13w = 'sylvania_CFL_13w'
ge_incandescant_25w = 'ge_incandescant_25w'
ge_incandescent_60w = 'ge_incandescant_60w'
philips_incandescent_40w = 'philips_incandescent_40w'
feit_led17p5w = 'feit_led17.5w'

master_brf_list = np.array([ecosmart_CFL_14w, maxlite_CFL_15w, sylvania_CFL_13w, ge_incandescant_25w, philips_incandescent_40w])

height = 576
width = 1024

savgol_window = 31

################################################################################################

#returns the mean value of an array
def mean(array):
    np_array = np.array(array)
    return np.sum(np_array)/len(np_array)

#returns the standard deviation of an array based on the mean
def std_dev(array, mean):
    sum_dif_squared = 0
    for value in array:
        sum_dif_squared += (value - mean)**2
    sum_dif_squared = sum_dif_squared/len(array)
    return math.sqrt(sum_dif_squared)
    
################################################################################################

#crops a brf from an image; 
def crop_brf(brf_img):
    return brf_img[639:1266,1110]

#aligns brfs by aligning phases
def align_brfs(brf_1, brf_2):
    #shfit amount corresponds to second argument (brf_2)
    shift_amount = phase_align(brf_1, brf_2, [10, 90]) #[10, 90] => region of interest; figure out what this is???
    shifted_brf_2 = shift(brf_2, shift_amount, mode = 'nearest')
    return brf_1, shifted_brf_2

#takes in two BRFs and truncates the longer one
def truncate_longer(brf_1, brf_2):
    if len(brf_1) > len(brf_2):
        brf_1 = brf_1[0:len(brf_2)]
    else:
        brf_2 = brf_2[0:len(brf_1)]
    return brf_1, brf_2

#finds the closest value in an array based on in the input parameter 'value'
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#returns the paths for rolling and dc image pairs; rolling returned first, then dc
def get_rolling_dc_paths(img_path):
    rolling = img_path + '_rolling.jpg'
    dc = img_path + '_dc.jpg'
    return rolling, dc

#gets the image from an input path
def img_from_path(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (width, height))
    return img

#shows the plot of a BRF/array with a specific figure size
def show_plot(array):
    plt.figure(figsize = (10, 1))
    plt.plot(array)
    plt.show()

#shows two BRFs on the same figure
def show_two_brfs(brf_1, brf_2):
    plt.figure(figsize = (10, 2))
    plt.plot(brf_1, label = '1')
    plt.plot(brf_2, label = '2')
    plt.legend()
    plt.show()

#plots a column every 100 pixels (width-wise)
def plot_entire_image(rolling_image):
    height = rolling_image.shape[0]
    width = rolling_image.shape[1]
    for col in range(0, width, 100):
        plt.plot(ss.savgol_filter(rolling_image[0:height, col], savgol_window, 3))
        plt.show()

#shows the image with cv2
def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#gets BRF; duplicate above
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

#normalized the BRF based on BRF min and max values
def normalize_brf(brf):
    brf_points = []
    # min = int(np.amin(brf))
    # max = int(np.amax(brf))
    min = np.amin(brf)
    max = np.amax(brf)
    #FIND A BETTER WAY TO DO THIS (USING NP ARRAY)
    for i in range(len(brf)):
        brf_points.append((brf[i] - min)/(max - min))
    brf_points = np.array(brf_points)
    return brf_points

#normalized BRF from known 'start' and 'end' values
def normalize(brf, start, end):
    min = int(start)
    max = int(end)
    if end < start:
        min = int(end)
        max = int(start)
    brf_points = np.array([])
    for i in range(len(brf)):
        normalized = (int(brf[i]) - min)/(max - min)
        brf_points = np.concatenate((brf_points, [normalized]), 0)
    return brf_points

#maps BRF to 'out_min' and 'out_max' based on brf min and max
def map_to_out(brf, out_min, out_max):
    new_brf = []
    brf = np.array(brf)
    in_min = np.amin(brf)
    in_max = np.amax(brf)
    for val in brf:
        mapped = (val - in_min)*(out_max - out_min)/(in_max - in_min) + out_min
        new_brf.append(mapped)
    return new_brf

#maps 'in_min' and 'in_max' to 'out_min' and 'out_max'
def map_in_to_out(waveform, in_min, in_max, out_min, out_max):
    new_brf = []
    brf = np.array(waveform)
    for val in brf:
        mapped = (val - in_min)*(out_max - out_min)/(in_max - in_min) + out_min
        new_brf.append(mapped)
    return new_brf

#shows peaks of a BRF with the BRF in a figure; distance for peaks set to 60
def show_peaks_with_brf(brf):
    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    peak_values = brf[peak_indices]
    plt.plot(peak_indices, peak_values, 'x')
    plt.plot(brf)
    plt.show()

#returns both min and max indices on a BRF
def return_extrema(brf):
    peak_indices = np.array(ss.find_peaks(brf, distance = 60)[0])
    nadir_indices = np.array(ss.find_peaks(-brf, distance = 60)[0])
    extrema_indices = sorted(np.concatenate((peak_indices, nadir_indices), 0))
    # extrema_values = brf[extrema_indices]
    return extrema_indices

#gets cycles from ONE BRF
def cycles_from_brf(brf, option):
    cycle_list = []
    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    for i in range(len(peak_indices)-1):
        cycle = brf[peak_indices[i]:peak_indices[i+1]]
        if option == 'normalize': cycle = normalize_brf(cycle)
        cycle_list.append(cycle)
    return cycle_list

#extracts and normalizes each half cycle based on the peak and nadir indices
def normalize_half_cycles(brf):
    half_cycles = []
    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    nadir_indices = ss.find_peaks(-brf, distance = 60)[0]
    extrema_indices = np.concatenate((np.array(peak_indices), np.array(nadir_indices)), 0)
    extrema_indices = sorted(extrema_indices)

    # extrema_values = brf[extrema_indices]
    for i in range(len(extrema_indices)-1):
        half_cycle = brf[extrema_indices[i]:extrema_indices[i+1]]
        half_cycle = normalize_brf(half_cycle)
        half_cycles.append(half_cycle)
    return half_cycles

#constructs a fitted sinusoid to the normalized brf
#removes sinusoidal wave from BRF
def remove_sin(norm_brf, name):
    fitted_sinusoid = np.array([]) #concatenate sinusoids to fit brf
    norm_brf = np.array(norm_brf)
    
    value_first = int(round(norm_brf[0]))
    value_last = int(round(norm_brf[len(norm_brf)-1]))
    
    brf = map_to_out(norm_brf, -1, 1) #mapping to between -1 and 1 to fit a sin wave
    brf = np.array(brf)

    start = np.pi/2
    stop = -np.pi/2

    if value_first == 0: #initial values need to be flipped if starting at 0
        start = -start
        stop = -stop

    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    nadir_indices = ss.find_peaks(-brf, distance = 60)[0]
    extrema_indices = sorted(np.concatenate((peak_indices, nadir_indices), 0))
    x = np.linspace(start, stop, extrema_indices[0]+1) #peak_indices[0] + 1 b/c brf starts at 0?
    fitted_sinusoid = np.sin(x)
    for i in range(len(extrema_indices)-1):
        start = -start
        stop = -stop
        x = np.linspace(start, stop, extrema_indices[i+1]-extrema_indices[i])
        fitted_sinusoid = np.concatenate((fitted_sinusoid, np.sin(x)), 0)

    x = np.linspace(-start, -stop, (len(brf)-1) - extrema_indices[len(extrema_indices)-1])
    fitted_sinusoid = np.concatenate((fitted_sinusoid, np.sin(x)), 0)

    if len(brf) != len(fitted_sinusoid): #TEMPORARY FIX
        fitted_sinusoid = np.concatenate((fitted_sinusoid, [value_last]), 0)

    new_brf = np.subtract(brf, fitted_sinusoid)

    return fitted_sinusoid

    fig, ax = plt.subplots(2,1)
    # fig.tight_layout(pad = 4.0)
    ax[0].plot(brf, label = 'Original BRF')
    ax[0].plot(fitted_sinusoid, label = 'Fitted Sinusoid')
    ax[0].set_title(name)
    ax[0].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax[1].plot(new_brf, label = 'New Subtracted BRF')
    ax[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.show()

    # plt.plot(brf, label = 'original')
    # plt.plot(fitted_sinusoid, label = 'fitted')
    # plt.plot(new_brf, label = 'subtracted')
    
    # x = np.arrange
    # plt.plot(np.sin())
    # plt.show()
    # print(f'First: {value_first}\nLast: {value_last}')

#returns normalized BRF via half cycles
def extract_normalized_brf(smoothed_brf):
    new_brf = np.array([])
    cycle_list = normalize_half_cycles(smoothed_brf) #extracts and normalizes each half cycle
    for cycle in cycle_list:
        new_brf = np.concatenate((new_brf, cycle), 0)
    return new_brf

################################################################################################

#pearson coefficient method
def pearson_coeff(brf_1, brf_2):
    if len(brf_1) > len(brf_2): brf_1 = brf_1[0:len(brf_2)]
    elif len(brf_2) > len(brf_1): brf_2 = brf_2[0:len(brf_1)]
    r, p = pearsonr(brf_1, brf_2)
    r = round(r, 2)
    return r

#cross correlation
def cross_corr(brf_1, brf_2):
    # correlate = ss.correlate(brf_1, brf_2, mode = 'full')/len(brf_1)
    correlate = np.correlate(brf_1, brf_2, mode = 'full')/len(brf_1)
    show_plot(correlate)
    correlate = map_in_to_out(correlate, -0.5, 0.5, 0, 1) #maps to 0 and 1
    show_plot(correlate)
    max = round(np.amax(np.array(correlate)), 6)
    return max

################################################################################################
#list-based methods

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

        # print(brf_path_1)
        # show_plot(brf_1)
        # print(brf_path_2)
        # show_plot(brf_2)

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

def process_brfs_from_list(brf_list_1, brf_list_2): #remove sin from each BRF
    list_1 = []
    list_2 = []

    for i in range(len(brf_list_1)):
        fitted_1 = fit_biased_sin(brf_list_1[i], master_brf_list[i])
        fitted_2 = fit_biased_sin(brf_list_2[i], master_brf_list[i])
        list_1.append(fitted_1)
        list_2.append(fitted_2)
    
    return list_1, list_2

def smooth_brfs_from_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []

    for i in range(len(brf_list_1)):
        smoothed_1 = ss.savgol_filter(brf_list_1[i], savgol_window, 3)
        smoothed_2 = ss.savgol_filter(brf_list_2[i], savgol_window, 3)
        # show_plot(smoothed_1)
        # show_plot(smoothed_2)
        list_1.append(smoothed_1) #window, polynomial order
        list_2.append(smoothed_2)

    return list_1, list_2

def normalize_smoothed_brf_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []
    normalized_1 = np.array([])
    normalized_2 = np.array([])
    for i in range(len(brf_list_1)):
        normalized_1 = map_to_out(extract_normalized_brf(brf_list_1[i]), -1, 1)
        normalized_2 = map_to_out(extract_normalized_brf(brf_list_2[i]), -1, 1)
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

################################################################################################

def fit_raw_brf(smoothed_1):
    extrema_indices = return_extrema(smoothed_1)

    norm_raw_brf_1 = np.array([])
    for i in range(len(extrema_indices)-1):
        temp = smoothed_1[extrema_indices[i]:extrema_indices[i+1]]
        norm_temp = normalize(temp, temp[0], temp[len(temp)-1])
        norm_raw_brf_1 = np.concatenate((norm_raw_brf_1, norm_temp), 0)        

    brf_cropped_1 = smoothed_1[extrema_indices[0]:extrema_indices[len(extrema_indices)-1]]
    normalized_cropped_1 = normalize_brf(brf_cropped_1)
    norm_raw_brf_1 = map_to_out(norm_raw_brf_1, -1, 1)

    fitted_1 = fit_biased_sin(smoothed_1, 'name')
    
    plt.plot(norm_raw_brf_1)
    plt.plot(fitted_1)
    plt.show()

#figure out what's best to input into this method
def fit_sinusoid(smoothed_brf): #input BRF needs to be smoothed ONLY; need to get extrema points
    #want extrema points for all in norm_smoothed brf
    all_extrema_points = np.array([]) #includes beginning and ending points

    zero_crossing_indices = np.array([])

    norm_smoothed = extract_normalized_brf(smoothed_brf)
    norm_smoothed = np.array(map_to_out(norm_smoothed, -1, 1))

    fitted_sinusoid = np.array([])

    #difference between all_extrema_points and extrema_indices: extrema_indices does not contain beginning and end points
    all_extrema_points = np.concatenate((all_extrema_points, [0]), 0)
    extrema_indices = return_extrema(norm_smoothed)
    all_extrema_points = np.concatenate((all_extrema_points, extrema_indices), 0)
    all_extrema_points = np.concatenate((all_extrema_points, [len(norm_smoothed)-1]), 0).astype(int)

    value_first = int(round(norm_smoothed[0]))
    value_last = int(round(norm_smoothed[len(norm_smoothed)-1]))
    # print(f'First: {value_first}\nLast: {value_last}')

    start = np.pi
    end = -np.pi

    show_plot(norm_smoothed)

    for i in range(0, len(all_extrema_points), 2):
        half_cycle = np.array(norm_smoothed[all_extrema_points[i]:all_extrema_points[i+1]])
        zero_crossing_indices = np.concatenate((zero_crossing_indices, np.where(half_cycle == find_nearest(half_cycle, 0))[0] + all_extrema_points[i]), 0)

    zero_crossing_indices = zero_crossing_indices.astype(int)
    # crossing_values = norm_smoothed[zero_crossing_indices]

    #start with quarter wave first
    if value_first == -1:
        x = np.linspace(-np.pi/2, 0, zero_crossing_indices[0])
        fitted_sinusoid = np.concatenate((fitted_sinusoid, np.sin(x)), 0)
    else:
        start = -np.pi
        end = np.pi
        x = np.linspace(np.pi/2, np.pi, zero_crossing_indices[0])
        fitted_sinusoid = np.concatenate((fitted_sinusoid, np.sin(x)), 0)

    for i in range(len(zero_crossing_indices)-1):
        sin_length = zero_crossing_indices[i+1] - zero_crossing_indices[i]
        x = np.linspace(start, end, sin_length)
        # x = x[1:len(x)-1]
        fitted_sinusoid = np.concatenate((fitted_sinusoid, np.sin(x)), 0)

    #need to figure out how to piece end together
    #--> depends on start and end (quarter wave for 3/4 wave)

    # plt.plot(zero_crossing_indices, crossing_values, 'x')
    plt.plot(fitted_sinusoid)
    plt.plot(norm_smoothed)
    plt.show()

#creates an unbiased sinusoid and aligns that with the normalized, smoothed BRF
def align_sinusoid(normalized_smoothed_brf, name):
    sinusoid = np.array([])
    brf = np.array(map_to_out(normalized_smoothed_brf, -1, 1))
    brf_peak_finding = brf

    avg_dist_bt_peaks = 0

    value_first = int(round(normalized_smoothed_brf[0]))
    value_last = int(round(normalized_smoothed_brf[len(normalized_smoothed_brf)-1]))
    
    #set these values if initial value starts at 1
    start = np.pi/2
    end = -np.pi/2*3

    #change if initial value is at low
    if value_first == -1:
        brf_peak_finding = -brf
        start = -start
        end = -end

    indices = ss.find_peaks(brf_peak_finding, distance = 60)[0]
    
    for i in range(len(indices)-1): #this step is to find the average period length
        dist = indices[i+1] - indices[i]
        avg_dist_bt_peaks += dist
    
    avg_dist_bt_peaks = int(round(avg_dist_bt_peaks/(len(indices)-1)))

    for i in range(len(indices)):
        x = np.linspace(start, end, avg_dist_bt_peaks)
        sinusoid = np.concatenate((sinusoid, np.sin(x)), 0)
    
    if value_last != value_first:
        x = np.linspace(start, end/3, int(round(avg_dist_bt_peaks/2)))
        sinusoid = np.concatenate((sinusoid, np.sin(x)), 0)

    wave_1, wave_2 = truncate_longer(brf, sinusoid)

    show_two_brfs(wave_1, wave_2)

    sub_1 = np.subtract(wave_1, wave_2)
    plt.plot(sub_1)
    plt.title(name)
    plt.show()

    # brf, aligned_sin = align_brfs(brf, sinusoid)

    # brf, aligned_sin = truncate_longer(brf, aligned_sin)

    # sub_2 = np.subtract(brf, aligned_sin)
    # show_plot(sub_2)

def align_nadirs(normalized_smoothed_brf):
    sinusoid = np.array([])
    brf = np.array(map_to_out(normalized_smoothed_brf, -1, 1))

    value_first = int(round(normalized_smoothed_brf[0]))
    value_last = int(round(normalized_smoothed_brf[len(normalized_smoothed_brf)-1]))

    nadir_indices = ss.find_peaks(-brf, distance = 60)[0]
    # nadir_values = brf[nadir_indices]

    if value_first == 1:
        x = np.linspace(np.pi/2, -np.pi/2, nadir_indices[0]+1) #this should not be +1?? DOUBLE CHECK
        sinusoid = np.concatenate((sinusoid, np.sin(x)), 0)
    else:
        x = np.linspace(-np.pi/2, np.pi/2*3, nadir_indices[0]+1)
        sinusoid = np.concatenate((sinusoid, np.sin(x)), 0)

    for i in range(len(nadir_indices)-1):
        x = np.linspace(-np.pi/2, np.pi/2*3, nadir_indices[i+1]-nadir_indices[i])
        sinusoid = np.concatenate((sinusoid, np.sin(x)), 0)

    if value_last == 1:
        x = np.linspace(-np.pi/2, np.pi/2, len(brf)-nadir_indices[len(nadir_indices)-1])
        sinusoid = np.concatenate((sinusoid, np.sin(x)))

    wave_1, wave_2 = truncate_longer(brf, sinusoid)

    show_two_brfs(wave_1, wave_2)

    subtracted_wave = np.subtract(wave_1, wave_2)

    show_plot(subtracted_wave)

################################################################################################

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

def pearson_correlation_heat_map(waveform_list_1, waveform_list_2):
    corr_list = []
    corr_heatmap = []

    #double checked with x/y-axis is correct; seems correct
    for waveform_1 in waveform_list_1:
        for waveform_2 in waveform_list_2:
            r = pearson_coeff(waveform_1, waveform_2)
            print(f'Pearson Coeff: {r}')
            show_two_brfs(waveform_1, waveform_2)
            corr_list.append(r)
        print()
        corr_list = np.flip(corr_list)
        corr_heatmap.append(corr_list)
        corr_list = []
    
    fig, ax = plt.subplots()
    # pcoeff_heatplot = ax.imshow(pcoeff_heatmap, cmap = "Blues")
    cross_corr_heatplot = ax.imshow(corr_heatmap, cmap = "Blues")
    ax.set_xticks(np.arange(len(waveform_list_1)))
    ax.set_yticks(np.arange(len(waveform_list_2)))
    ax.set_xticklabels(np.flip(np.arange(len(waveform_list_1))))

    # ax.set_yticklabels(brf_name_list_1)
    # ax.set_title(title)

    for i in range(len(corr_heatmap)):
        for j in range(len(corr_heatmap[i])):
            text = ax.text(j, i, corr_heatmap[i][j],
                        ha="center", va="center", color="black")

    colorbar = fig.colorbar(cross_corr_heatplot)
    # plt.xticks(rotation = 45)
    fig.tight_layout()
    plt.show()

################################################################################################

def check_heatmap_direction():
    x = np.array([1,2,3,4,5])
    y = np.array([1,2,3,4,5])
    mult_row = []
    mult_heatmap = []

    for i in x:
        for j in y:
            mult_row.append(i*j)
        mult_row = np.array(mult_row)
        mult_row = np.flip(mult_row)
        mult_heatmap.append(mult_row)
        mult_row = []
    
    fig, ax = plt.subplots()
    heatplot = ax.imshow(mult_heatmap, cmap = 'Blues')
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticklabels(np.flip(x))
    ax.set_yticklabels(y)

    for i in range(len(mult_heatmap)):
        for j in range(len(mult_heatmap[i])):
            text = ax.text(j, i, mult_heatmap[i][j],
                        ha="center", va="center", color="black")

    colorbar = fig.colorbar(heatplot)
    # plt.xticks(rotation = 45)
    fig.tight_layout()
    plt.show()

def test_cross_corr(num_periods_1, num_periods_2): #using two sinusoids
    sinusoid_1 = np.array([])
    sinusoid_2 = np.array([])

    for i in range(num_periods_1):
        x = np.linspace(np.pi, -np.pi, 200)
        sinusoid_1 = np.concatenate((sinusoid_1, np.sin(x)), 0)
        np.delete(sinusoid_1, -1)

    for i in range(num_periods_2):
        x = np.linspace(np.pi, -np.pi, 200)
        sinusoid_2 = np.concatenate((sinusoid_2, np.sin(x)), 0)
        np.delete(sinusoid_2, -1)

    show_plot(sinusoid_1)
    show_plot(sinusoid_2)

    correlate = ss.correlate(sinusoid_2, sinusoid_1, mode = 'full')/len(sinusoid_1)

    show_plot(correlate)

def t_test(array_1, array_2):
    mean_1 = mean(array_1)
    mean_2 = mean(array_2)
    std_dev_1 = std_dev(array_1, mean_1)
    std_dev_2 = std_dev(array_2, mean_2)
    denominator = math.sqrt(std_dev_1**2/len(array_1) + std_dev_2**2/len(array_2))
    t = abs(mean_1 - mean_2)/denominator
    return t

################################################################################################
#frequency methods

#find average length of cycle
#need to normalize so that all 120Hz amplitudes are the same?
def return_fft(normalized_smoothed_brf):
    peak_indices = ss.find_peaks(normalized_smoothed_brf, distance = 60)[0]
    avg_cycle_len = 0
    for i in range(len(peak_indices)-1):
        avg_cycle_len += peak_indices[i+1] - peak_indices[i]
    avg_cycle_len = int(round(avg_cycle_len/(len(peak_indices)-1)))

    sample_rate = avg_cycle_len * 120
    num_samples = len(normalized_smoothed_brf)
    # print(num_samples)
    sample_spacing = 1/sample_rate
    # print(sample_spacing)
    xf = np.linspace(0, 1.0/(2*sample_spacing), num_samples//2)
    # print(xf)
    yf = fft(normalized_smoothed_brf)
    yf = 2.0/num_samples*np.abs(yf[0:num_samples//2])
    yf = normalize_brf(yf) #need to change the name of this method

    
    # peak_indices = ss.find_peaks(yf)[0]
    # peak_values = yf[peak_indices]
    # max_peak = np.amax(peak_values)
    # print(max_peak)
    # index = np.where(yf == max_peak)
    # print(xf[index])

    # plt.plot(xf, yf)
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # # plt.xticks(np.arange(0, 1/(sample_spacing*2), 100))
    # plt.title('Sylvania CFL')
    # plt.grid()
    # plt.show()

    return xf, yf

#input fft of brf
def remove_120hz(yf):
    max_peak = np.amax(yf)
    max_peak_index = np.where(yf == max_peak)[0]
    nadir_indices = ss.find_peaks(yf)[0]
    for nadir_freq in nadir_indices:
        if nadir_freq > max_peak_index:
            yf[0:nadir_freq] = 0
            #interpolate points after??
            break
    return yf

#return 5 highest peak frequencies (lowest freq to highest)
#this method will not work for freq components that are not decending in strength
def peak_frequencies(xf, yf, num_freq_peaks):
    n_top_freq_peak_indices = []
    smoothed_yf = ss.savgol_filter(yf, 9, 2) #poly => 2?
    peak_indices = ss.find_peaks(smoothed_yf)[0] #indices for smoothed_yf peaks
    smoothed_xf_freq_values = xf[peak_indices] #freq values corresponding to smoothed_yf peaks

    yf_peak_indices = ss.find_peaks(yf, width = 1)[0] #peak indices for yf (original FFT)
    indices = yf_peak_indices[0:num_freq_peaks] #showing top num_freq_peaks (number of) peaks
    yf_peak_values = yf[indices]
    xf_freq_values = xf[indices]
    # yf_peak_values = yf[yf_peak_indices] #peak values
    # xf_freq_values = xf[yf_peak_indices] #freq values corresponding to peaks

    for i in range(len(xf_freq_values)):
        xf_freq_values[i] = round(xf_freq_values[i], 2)

    print(f'Frequency Values: {xf_freq_values}')
    print(f'Peak Values: {yf_peak_values}')

    plt.plot(xf_freq_values, yf_peak_values, 'x')
    plt.plot(xf, yf)
    plt.show()

    '''
    #loop through and get each peak value index
    for i in range(num_freq_peaks):
        #get closest corresponding frequency value from smoothed_xf --> original xf
        corresponding_freq_value = find_nearest(xf_freq_values, smoothed_xf_freq_values[i])
        index_value = int(np.where(xf == corresponding_freq_value)[0]) #get the index of that corresponding frequency value
        index_in_yf_peak_indices_array = int(np.where(yf_peak_indices == index_value)[0]) #index that corresponds to position in yf_peak_indices ARRAY

        # check if following peak value is greater
        if yf_peak_values[index_in_yf_peak_indices_array+1] > yf_peak_values[index_in_yf_peak_indices_array]:
            n_top_freq_peak_indices.append(yf_peak_indices[index_in_yf_peak_indices_array+1])
        else:
            n_top_freq_peak_indices.append(index_value)
    
    peak_freq_values = yf[n_top_freq_peak_indices]
    freq_values = xf[n_top_freq_peak_indices]
    for i in range(len(freq_values)):
        freq_values[i] = round(freq_values[i], 2)

    print(freq_values)

    plt.plot(freq_values, peak_freq_values, 'x')
    plt.plot(xf, yf)
    plt.show()
    '''

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

################################################################################################

def save_brf_csv(brf_array, bulb):
    save_path = ''
    brf_file_name = ''
    if bulb == 'sylvania':
        save_path = brf_save_load_path + '\\' + sylvania_CFL_13w
        brf_file_name = sylv_file_num + '.csv'
    elif bulb == 'philips':
        save_path = brf_save_load_path + '\\' + philips_incandescent_40w
        brf_file_name = philips_file_num + '.csv'
    os.chdir(save_path)
    # num_files = len(os.listdir(save_path))
    np.savetxt(brf_file_name, brf_array, delimiter = ',')

def load_brfs(bulb):
    brf_list = []
    load_path = ''
    if bulb == 'sylvania': load_path = brf_save_load_path + '\\' + sylvania_CFL_13w
    if bulb == 'philips': load_path = brf_save_load_path + '\\' + philips_incandescent_40w
    os.chdir(load_path)
    num_files = len(os.listdir(load_path))
    for i in range(num_files):
        file_name = bulb + '_' + str(i) + '.csv'
        brf = np.genfromtxt(file_name, delimiter = ',')
        brf_list.append(brf)
    return brf_list

def load_scope_waveforms(path):
    os.chdir(path)
    num_files = len(os.listdir(path))
    for i in range(num_files):
        file_name = path + '\\' + 'waveform_' + str(i) + '.csv'
        df = pd.read_csv(file_name)
        brf = df.iloc[:,1]
        # print(brf)
        return_fft(brf)
        # show_plot(brf)

################################################################################################

if __name__ == '__main__':
    # path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_cfl_13w'
    # load_scope_waveforms(path)
    '''
    #retrieves raw brfs
    brf_list_1, brf_list_2 = extract_brfs_from_list(master_brf_list)
    #smooths raw brfs
    smoothed_list_1, smoothed_list_2 = smooth_brfs_from_list(brf_list_1, brf_list_2)
    #normalized between 0 and 1
    norm_smooth_list_1, norm_smooth_list_2 = normalize_brfs_from_list(smoothed_list_1, smoothed_list_2)

    norm_smooth_list_1, norm_smooth_list_2 = normalize_smoothed_brf_list(smoothed_list_1, smoothed_list_2)

    for i in range(len(norm_smooth_list_1)):
        xf_1, yf_1 = return_fft(norm_smooth_list_1[i])
        xf_2, yf_2 = return_fft(norm_smooth_list_2[i])
        print(master_brf_list[i])
        peak_frequencies(xf_1, yf_1, 5)
        peak_frequencies(xf_2, yf_2, 5)

    #each brf in each list is normalized and smoothed
    brf_list_1 = load_brfs('philips') #if things aren't working, check if brfs are being correctly saved/loaded
    brf_list_2 = load_brfs('sylvania')

    fft_list_1 = []
    fft_list_2 = []

    for i in range(len(brf_list_1)):
        xf_1, yf_1 = return_fft(brf_list_1[i])
        xf_2, yf_2 = return_fft(brf_list_2[i])
        # fft_list_1.append([xf_1, remove_120hz(yf_1)])
        # fft_list_2.append([xf_2, remove_120hz(yf_2)])
        # fft_list_1.append([xf_1, yf_1])
        fft_list_2.append([xf_2, yf_2])

        peak_frequencies(xf_1, yf_1, 5)
        peak_frequencies(xf_2, yf_2, 5)

    # for fft_1 in fft_list_1:
    #     for fft_2 in fft_list_2:
    #         print(t_test(fft_1, fft_2))

    fft_list = fft_list_1 + fft_list_2

    for fft_1 in fft_list:
        for fft_2 in fft_list:
            print(t_test(fft_1[1], fft_2[1]))
            # plt.plot(fft_1[0], fft_1[1])
            # plt.plot(fft_2[0], fft_2[1])
            plt.plot(fft_1[1])
            plt.plot(fft_2[1])
            plt.show()
        print()
    '''
    # pearson_correlation_heat_map(fft_list, fft_list)

    # img_1 = img_from_path(ecosmart_CFL)
    img_1 = img_from_path(subtr_path)
    # img_2 = img_from_path(philips_incandescent)
    # img_3 = img_from_path(sylvania_CFL)

    brf_1 = crop_brf(img_1)
    show_plot(brf_1)
    save_brf_csv(brf_1, 'sylvania')
    '''
    smoothed_1 = ss.savgol_filter(brf_1, savgol_window, 3)

    norm_smoothed_1 = extract_normalized_brf(smoothed_1)
    norm_smoothed_1 = map(norm_smoothed_1, -1, 1)
    save_brf_csv(norm_smoothed_1, 'philips')
    '''

    # fig, ax = plt.subplots(3,1)
    # # fig.tight_layout(pad = 4.0)
    # ax[0].plot(brf_1, label = 'Raw BRF')
    # ax[0].set_title('Ecosmart CFL 14W')
    # ax[0].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    # box = ax[0].get_position()
    # ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # ax[1].plot(smoothed_1, label = 'Smoothed BRF')    
    # ax[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    # box = ax[1].get_position()
    # ax[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # ax[2].plot(normalized_1, label = 'Normalized BRF')
    # extrema_indices, extrema_values = return_extrema(normalized_1)
    # ax[2].plot(extrema_indices, extrema_values, 'x')
    # ax[2].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    # box = ax[2].get_position()
    # ax[2].set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # cycle_list_1, cycle_list_2 = extract_cycles_from_list(smoothed_list_1, smoothed_list_2)
    # cycle_list_1, cycle_list_2 = extract_cycles_from_list(norm_smooth_list_1, norm_smooth_list_2)
    # normalized_cycle_list_1, normalized_cycle_list_2 = normalize_brf_list(cycle_list_1, cycle_list_2)

    # correlation_heat_map(brf_list_1, brf_list_2, 'Raw BRF Cross Correlation Heat Map')
    # correlation_heat_map(smoothed_list_1, smoothed_list_2, 'Smoothed BRF Cross Correlation Heat Map')
    # correlation_heat_map(norm_smooth_list_1, norm_smooth_list_2, 'Normalized-Smoothed BRF Cross Correlation Heat Map')
    # correlation_heat_map(smoothed_list_1, smoothed_list_2, 'Cycle Cross Corrlation Heat Map') #??
    # correlation_heat_map(cycle_list_1, cycle_list_2, 'Averaged Cycle Cross Correlation Heat Map')
    # correlation_heat_map(normalized_cycle_list_1, normalized_cycle_list_2, 'Normalized-Averaged Cycle Cross Correlation Heat Map')