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
path_2 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\2'
path_3 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\3'
path_4 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1'
# ecosmart_blurred = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\ecosmart_blurred.jpg'
# philips_uncalibrated = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\philips_uncalibrated.jpg'
# philips_calibrated = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\philips_calibrated.jpg'
ecosmart_CFL = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\ecosmart_CFL_14w_0_rolling.jpg'
philips_incandescent = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\philips_incandescent_40w_0_rolling.jpg'
sylvania_CFL = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images\blurred_1\sylvania_CFL_13w_0_rolling.jpg'
#make classes man..

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

def extract_normalized_brf(brf):
    new_brf = np.array([])
    cycle_list = cycles_from_brf(brf, 'normalize')
    for cycle in cycle_list:
        new_brf = np.concatenate((new_brf, cycle), 0)
    show_plot(new_brf)

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

def crop_brf(brf, start_index, end_index):
    return brf[start_index:end_index]

def align_brfs(brf_1, brf_2):
    #shfit amount corresponds to second argument (brf_2)
    shift_amount = phase_align(brf_1, brf_2, [10, 90]) #[10, 90] => region of interest; figure out what this is???
    shifted_brf_2 = shift(brf_2, shift_amount, mode = 'nearest')
    return brf_1, shifted_brf_2

def pearson_coeff(brf_1, brf_2):
    # array = np.array([brf_1, brf_2])
    # array = array.reshape(len(brf_1), len(array))
    # df = pd.DataFrame(array)

    # overall_pearson_r = df.corr(method = 'pearson').iloc[0,1]
    # print(f"Pandas computed Pearson r: {overall_pearson_r}")
    # # out: Pandas computed Pearson r: 0.2058774513561943
    # print(len(brf_1))
    # print(len(brf_2))


    if len(brf_1) > len(brf_2): brf_1 = brf_1[0:len(brf_2)]
    elif len(brf_2) > len(brf_1): brf_2 = brf_2[0:len(brf_1)]

    # print(len(brf_1))
    # print(len(brf_2))

    r, p = pearsonr(brf_1, brf_2)
    # print(f"Scipy computed Pearson r: {r} and p-value: {p}")
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
        brf_1 = normalize_brf(img_1[770:1200,550])
        brf_2 = normalize_brf(img_2[640:1000,2220])

        brf_list_1.append(brf_1)
        brf_list_2.append(brf_2)
    
    return brf_list_1, brf_list_2

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

#avg_window is the number of cycles I want to average
def extract_cycles_from_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []
    for i in range(len(brf_list_1)):
        list_1.append(average_periods(brf_list_1[i], 10))
        list_2.append(average_periods(brf_list_2[i], 10))
    return list_1, list_2

def normalize_brf_list(brf_list_1, brf_list_2):
    list_1 = []
    list_2 = []
    for i in range(len(brf_list_1)):
        list_1.append(normalize_brf(brf_list_1[i]))
        list_2.append(normalize_brf(brf_list_2[i]))
    return list_1, list_2

def cycles_from_brf(brf, option): #gets cycles from ONE BRF
    cycle_list = []
    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    for i in range(len(peak_indices)-1):
        cycle = brf[peak_indices[i]:peak_indices[i+1]]
        if option == 'normalize': cycle = normalize_brf(cycle)
        cycle_list.append(cycle)
    return cycle_list

#using a cheesy method to find the peaks
def average_periods(brf, num_cycles_avgerages):
    min_cycle_len = 9999
    cycle_list = []
    cycle_len_list = []
    aligned_cycle_list = []
    cycle_avg = 0

    '''
    filtering for peaks doesn't entirely work..
    '''

    # avg_dist_bt_peaks = 0
    # not_finished = True

    # #first filter for initial false positive peaks
    # peak_indice_approximations = ss.find_peaks(brf)[0] #need to know approximate period distance; maybe a problem
    # peak_indice_approximation_values = brf[peak_indice_approximations]
    # interp_model = interpolate.interp1d(peak_indice_approximations, peak_indice_approximation_values, fill_value='extrapolate')
    # interpolated_brf = np.zeros(len(brf))
    # interpolated_brf[peak_indice_approximations] = peak_indice_approximation_values
    # filtered_peak_brf = interp_model(np.arange(0, len(brf), 1))
    # peak_indices = ss.find_peaks(filtered_peak_brf)[0]

    # #find average distance between peaks
    # for i in range(len(peak_indices)-1):
    #     avg_dist_bt_peaks += peak_indices[i+1] - peak_indices[i]
    # avg_dist_bt_peaks /= (len(peak_indices)-1)
    # print(avg_dist_bt_peaks)
    # #remove false positives between peaks
    # while not_finished:
    #     # print(len(peak_indices)-1)
    #     for i in range(len(peak_indices)-1):
    #         diff = peak_indices[i+1]-peak_indices[i]
    #         # print(diff)
    #         if diff < avg_dist_bt_peaks:
    #             index = np.where(peak_indices == peak_indices[i+1])
    #             peak_indices = np.delete(peak_indices, index[0])
    #             # peak_indices.remove(peak_indices[i+1])
    #             break
    #         if i == (len(peak_indices)-2):
    #             not_finished = False

    peak_indices = ss.find_peaks(brf, distance = 60)[0]
    # peak_values = brf[peak_indices]

    for i in range(len(peak_indices)-1):
        cycle = brf[peak_indices[i]:peak_indices[i+1]]
        cycle_list.append(cycle)

    aligned_cycle_list.append(cycle_list[0]) #append the first cycle in the list
    cycle_len_list.append(len(cycle_list[0]))

    #cross correlation of cycles to align cycles
    for i in range(len(cycle_list)-1):
        # shorter_cycle = None
        # longer_cycle = None
        # if len(cycle_list[i]) > len(cycle_list[i+1]):
        #     shorter_cycle = cycle_list[i+1]
        #     longer_cycle = cycle_list[i]
        # else:
        #     shorter_cycle = cycle_list[i]
        #     longer_cycle = cycle_list[i+1]
        # correlated_graph = ss.correlate(shorter_cycle, longer_cycle, mode = 'full')/len(shorter_cycle)
        # print(len(correlated_graph))
        # print(len(shorter_cycle))
        # print(len(longer_cycle))
        # show_plot(correlated_graph)
        # correlated_graph = np.array(correlated_graph[len(cycle_list[i]):(len(correlated_graph)-len(cycle_list[i]))])
        # show_plot(correlated_graph)
        # max = np.amax(correlated_graph)
        # index_max = np.where(correlated_graph == max)
        # plt.plot(correlated_graph)
        # plt.plot(cycle_list[i])
        # plt.plot(cycle_list[i+1])
        # plt.plot(index_max, max)
        # plt.show()

        aligned_brf_1, aligned_brf_2 = align_brfs(cycle_list[i], cycle_list[i+1])
        aligned_cycle_list.append(aligned_brf_2)
        cycle_len_list.append(len(aligned_brf_2))

    #sorts list according to length of cycles; [0] is cycle len, [1] is cycle
    sorted_max_cycle_len = sorted(list(zip(cycle_len_list,aligned_cycle_list)), key = lambda x: x[0], reverse = True)

    min_cycle_len = sorted_max_cycle_len[num_cycles_avgerages][0] #gets smallest cycle length in the x number (e.g. 10) of cycles to verage

    for i in range(num_cycles_avgerages):
        cycle_avg += sorted_max_cycle_len[i][1][0:min_cycle_len] #truncates all cycle lengths to minimum cycle length size
    cycle_avg /= num_cycles_avgerages

    return cycle_avg

#make another method that doesn't normalize
#uses smoothed BRFs
def cycle_cross_corr(brf_1, brf_2):
    xcorr_avgs = []
    new_brf = np.array([])
    cycle_list_1 = cycles_from_brf(brf_1, 'normalize')
    cycle_list_2 = cycles_from_brf(brf_2, 'normalize')
    for cycle in cycle_list_2:
        new_brf = np.concatenate((new_brf, cycle), 0)
    for cycle in cycle_list_1:
        corr = ss.correlate(cycle, new_brf, mode = 'full')/len(cycle)
        corr = corr[len(cycle):len(corr)-len(cycle)]
        peak_indices = ss.find_peaks(corr, distance = 50)[0]
        peak_values = corr[peak_indices]
        peak_avg = np.sum(peak_values)/len(peak_values)
        xcorr_avgs.append(peak_avg)
    corr_coeff = np.sum(np.array(xcorr_avgs))/len(xcorr_avgs)
    return corr_coeff

#name for list 2 will be reverse of name for list 1
def correlation_heat_map(brf_list_1, brf_list_2, title):
    peak_cross_corr_list = []
    cross_corr_name_list = []
    cross_corr_heatmap = []

    brf_name_list_1 = np.array(master_brf_list)
    brf_name_list_2 = np.flip(brf_name_list_1)

    #double checked with x/y-axis is correct; seems correct
    for brf_1 in brf_list_1:
        for brf_2 in brf_list_2:
            # pcoeff_list.append(round(pearson_coeff(brf_1, brf_2), 3))
            correlate = ss.correlate(brf_1, brf_2, mode = 'full')/len(brf_1)
            # correlate = cycle_cross_corr(brf_1, brf_2)
            max = round(np.amax(correlate), 3)
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
    # ax.set_title("Pearson Correlation Heat Map")
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
    # img_1 = img_from_path(sylvania_CFL)
    # img_2 = img_from_path(philips_incandescent)
    # img_3 = img_from_path(ecosmart_CFL)

    # brf_1 = brf_extraction(img_1)
    # brf_2 = brf_extraction(img_2)
    # brf_3 = brf_extraction(img_3)

    # smoothed_1 = ss.savgol_filter(brf_1, 51, 3)
    # smoothed_2 = ss.savgol_filter(brf_2, 51, 3)
    # smoothed_3 = ss.savgol_filter(brf_3, 51, 3)

    # normalized_1 = normalize_brf(smoothed_1)
    # normalized_2 = normalize_brf(smoothed_2)
    # normalized_3 = normalize_brf(smoothed_3)

    # max_1 = cross_corr(normalized_1, normalized_2)
    # max_2 = cross_corr(normalized_2, normalized_3)
    # max_3 = cross_corr(normalized_1, normalized_3)

    # print(max_1)
    # print(max_2)
    # print(max_3)

    brf_list_1, brf_list_2 = extract_brfs_from_list(master_brf_list)
    smoothed_list_1, smoothed_list_2 = smooth_brfs_from_list(brf_list_1, brf_list_2)
    # norm_smooth_list_1, norm_smooth_list_2 = normalize_smoothed_brf_list(smoothed_list_1, smoothed_list_2)
    # cycle_list_1, cycle_list_2 = extract_cycles_from_list(smoothed_list_1, smoothed_list_2)
    # cycle_list_1, cycle_list_2 = extract_cycles_from_list(norm_smooth_list_1, norm_smooth_list_2)
    # normalized_cycle_list_1, normalized_cycle_list_2 = normalize_brf_list(cycle_list_1, cycle_list_2)


    correlation_heat_map(brf_list_1, brf_list_2, 'Raw BRF Cross Correlation Heat Map')
    correlation_heat_map(smoothed_list_1, smoothed_list_2, 'Smoothed BRF Cross Correlation Heat Map')
    # correlation_heat_map(norm_smooth_list_1, norm_smooth_list_2, 'Normalized-Smoothed BRF Cross Correlation Heat Map')
    # # correlation_heat_map(smoothed_list_1, smoothed_list_2, 'Cycle Cross Corrlation Heat Map') #??
    # correlation_heat_map(cycle_list_1, cycle_list_2, 'Averaged Cycle Cross Correlation Heat Map')
    # correlation_heat_map(normalized_cycle_list_1, normalized_cycle_list_2, 'Normalized-Averaged Cycle Cross Correlation Heat Map')