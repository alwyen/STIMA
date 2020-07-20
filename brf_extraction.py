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

#make classes man..

ecosmart_CFL_14w = 'ecosmart_CFL_14w'
maxlite_CFL_15w = 'maxlite_CFL_15w'
sylvania_CFL_13w = 'sylvania_CFL_13w'
ge_incandescant_25w = 'ge_incandescant_25w'
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

def crop_brf(brf, start_index, end_index):
    return brf[start_index:end_index]

def align_brfs(brf_1, brf_2):
    #shfit amount corresponds to second argument (brf_2)
    shift_amount = phase_align(brf_1, brf_2, [10, 90]) #[10, 90] => region of interest; figure out what this is???
    shifted_brf_2 = shift(brf_2, shift_amount, mode = 'nearest')
    return brf_1, shifted_brf_2

def smooth_brf(bulb, path, window_size):
    # bulb_path = path + '\\' + bulb + ' '
    pass

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

def extract_brfs_from_list(master_brf_list):
    brf_list_1 = []
    brf_list_2 = []

    for bulb in master_brf_list:
        brf_path_1 = path_3 + '\\' + bulb + '_0_rolling.jpg'
        brf_path_2 = path_3 + '\\' + bulb + '_1_rolling.jpg'

        img_1 = img_from_path(brf_path_1)
        img_2 = img_from_path(brf_path_2)

        brf_1 = normalize_brf(brf_extraction(img_1))
        brf_2 = normalize_brf(brf_extraction(img_2))

        brf_list_1.append(brf_1)
        brf_list_2.append(brf_2)
    
    return brf_list_1, brf_list_2

def smooth_brfs_from_list(brf_list_1, brf_list_2):
    list_1 = np.copy(brf_list_1)
    list_2 = np.copy(brf_list_2)

    for i in range(len(list_1)):
        list_1[i] = ss.savgol_filter(list_1[i], 61, 3) #window, polynomial order
        list_2[i] = ss.savgol_filter(list_2[i], 61, 3)

    return list_1, list_2

#using a cheesy method to find the peaks
def average_periods(brf, avg_window):
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
        cycle_list.append(brf[peak_indices[i]:peak_indices[i+1]])

    aligned_cycle_list.append(cycle_list[0])
    cycle_len_list.append(len(cycle_list[0]))

    for i in range(len(cycle_list)-1):
        aligned_brf_1, aligned_brf_2 = align_brfs(cycle_list[i], cycle_list[i+1])
        aligned_cycle_list.append(aligned_brf_2)
        cycle_len_list.append(len(aligned_brf_2))

    sorted_max_cycle_len = sorted(list(zip(cycle_len_list,aligned_cycle_list),), key = lambda x: x[0], reverse = True)
    min_cycle_len = sorted_max_cycle_len[avg_window][0] #gets avg_window'th smallest cycle length

    for i in range(avg_window):
        cycle_avg += sorted_max_cycle_len[i][1][0:min_cycle_len] #truncates all cycle lengths to minimum cycle length size
    cycle_avg /= min_cycle_len
    # plt.plot(cycle_avg)
    # plt.show()

    return cycle_avg

#THIS CODE IS REALLY REALLY MESSY
def return_average_periods(bulb_path, path, savgov_window, avg_window):
    bulb_path_1 = path + '\\' + bulb_path + '_0'
    bulb_path_2 = path + '\\' + bulb_path + '_1'

    brf_1 = process_extract_brf(bulb_path_1)
    brf_2 = process_extract_brf(bulb_path_2)

    smoothed_brf_1 = ss.savgol_filter(brf_1, savgov_window, 3)
    smoothed_brf_2 = ss.savgol_filter(brf_2, savgov_window, 3)

    averaged_brf_1 = average_periods(smoothed_brf_1, avg_window)
    averaged_brf_2 = average_periods(smoothed_brf_2, avg_window)

    return averaged_brf_1, averaged_brf_2

def compare_brfs_same_bulb(bulb_path, path, savgov_window):
    bulb_path_1 = path + '\\' + bulb_path + '_0'
    bulb_path_2 = path + '\\' + bulb_path + '_1'

    brf_1 = process_extract_brf(bulb_path_1)
    brf_2 = process_extract_brf(bulb_path_2)

    smoothed_brf_1 = ss.savgol_filter(brf_1, savgov_window, 3)
    smoothed_brf_2 = ss.savgol_filter(brf_2, savgov_window, 3)
    # smoothed_brf_1 = brf_1
    # smoothed_brf_2 = brf_2

    aligned_brf_1, aligned_brf_2 = align_brfs(smoothed_brf_1, smoothed_brf_2)
    # show_two_brfs(aligned_brf_1, aligned_brf_2)

    cropped_brf_1 = crop_brf(smoothed_brf_1, 0, 250)
    correlate = ss.correlate(cropped_brf_1, smoothed_brf_2, mode = 'full')/len(cropped_brf_1)
    correlate = crop_brf(correlate, len(cropped_brf_1), len(correlate) - len(cropped_brf_1))
    return correlate

def compare_averaged_brfs(brf_1, brf_2):
    brf_1 = normalize_brf(brf_1)
    brf_2 = normalize_brf(brf_2)
    # show_plot(brf_1)
    # show_plot(brf_2)

    if len(brf_1) > len(brf_2): brf_1 = brf_1[0:len(brf_2)]
    elif len(brf_2) > len(brf_1): brf_2 = brf_2[0:len(brf_1)]

    # correlate = ss.correlate(brf_1, brf_2, mode = 'full')/len(brf_1)
    correlate = np.correlate(brf_1, brf_2, 'full')/len(brf_1)
    correlate = np.array(correlate)
    # show_plot(correlate)
    peak = np.amax(correlate)
    return peak

def compare_brfs(bulb_1, path_1, bulb_2, path_2, savgov_window):
    bulb_1 = path_1 + '\\' + bulb_1 + '_0'
    bulb_2 = path_2 + '\\' + bulb_2 + '_0'
    brf_rolling_1 = process_extract_brf(bulb_1)
    brf_rolling_2 = process_extract_brf(bulb_2)

    # brf_rolling_1 = crop_brf(brf_extraction(normalized_1), 0, 500)
    # brf_rolling_2 = crop_brf(brf_extraction(normalized_2), 0, 500)

    smoothed_brf_1 = ss.savgol_filter(brf_rolling_1, savgov_window, 3)
    smoothed_brf_2 = ss.savgol_filter(brf_rolling_2, savgov_window, 3)
    # smoothed_brf_1 = brf_rolling_1
    # smoothed_brf_2 = brf_rolling_2

    aligned_brf_1, aligned_brf_2 = align_brfs(smoothed_brf_1, smoothed_brf_2)

    cropped_brf_1 = crop_brf(smoothed_brf_1, 0, 250)
    correlate_1 = ss.correlate(cropped_brf_1, aligned_brf_2, mode = 'full')/len(cropped_brf_1)
    correlate_1 = crop_brf(correlate_1, len(cropped_brf_1), len(correlate_1) - len(cropped_brf_1))
    return correlate_1

def compare_different_sensitivity_brfs(brf_1, brf_2):
    img_path_1 = path_1 + '\\' + brf_1 + '_rolling.jpg'
    img_path_2 = path_2 + '\\' + brf_2 + '_0_rolling.jpg'
    brf_1 = brf_extraction(img_from_path(img_path_1))
    brf_2 = brf_extraction(img_from_path(img_path_2))
    smoothed_brf_1 = ss.savgol_filter(brf_1, 61, 3)
    smoothed_brf_2 = ss.savgol_filter(brf_2, 61, 3)
    normalized_brf_1 = normalize_brf(smoothed_brf_1)
    normalized_brf_2 = normalize_brf(smoothed_brf_2)
    aligned_brf_1, aligned_brf_2 = align_brfs(normalized_brf_1, normalized_brf_2)
    show_two_brfs(aligned_brf_1, aligned_brf_2)

#name for list 2 will be reverse of name for list 1
def correlation_heat_map(brf_list_1, brf_list_2):
    peak_cross_corr_list = []
    cross_corr_name_list = []
    cross_corr_heatmap = []

    brf_name_list_1 = np.array(master_brf_list)
    brf_name_list_2 = np.flip(brf_name_list_1)

    for brf_1 in brf_list_1:
        for brf_2 in brf_list_2:
            # pcoeff_list.append(round(pearson_coeff(brf_1, brf_2), 3))
            peak_cross_corr_list.append(round(compare_averaged_brfs(brf_1, brf_2), 3))
        # pcoeff_heatmap.append(pcoeff_list)
        cross_corr_heatmap.append(peak_cross_corr_list)
        # pcoeff_list = []
        peak_cross_corr_list = []
    
    fig, ax = plt.subplots()
    # pcoeff_heatplot = ax.imshow(pcoeff_heatmap, cmap = "Blues")
    cross_corr_heatplot = ax.imshow(cross_corr_heatmap, cmap = "Blues")
    ax.set_xticks(np.arange(len(brf_list_1)))
    ax.set_yticks(np.arange(len(brf_list_2)))
    ax.set_xticklabels(brf_name_list_2)
    ax.set_yticklabels(brf_name_list_1)
    # ax.set_title("Pearson Correlation Heat Map")
    ax.set_title('Averaged Cycle Cross Correlation Heat Map')

    for i in range(len(cross_corr_heatmap)):
        for j in range(len(cross_corr_heatmap[i])):
            text = ax.text(j, i, cross_corr_heatmap[i][j],
                        ha="center", va="center", color="black")

    colorbar = fig.colorbar(cross_corr_heatplot)
    plt.xticks(rotation = 45)
    fig.tight_layout(pad = 4.0)
    plt.show()

if __name__ == '__main__':
    brf_list_1, brf_list_2 = extract_brfs_from_list(master_brf_list)

    correlation_heat_map(brf_list_1, brf_list_2)

    smoothed_list_1, smoothed_list_2 = smooth_brfs_from_list(brf_list_1, brf_list_2)

    correlation_heat_map(smoothed_list_1, smoothed_list_2)

    # brf_name_list_1 = ['ecosmart_CFL', 'maxlite_CFL', 'sylvania_CFL', 'ge_incandescent', 'philips_incandescent']
    # brf_name_list_1.reverse()
    # brf_name_list_2 = ['ecosmart_CFL', 'maxlite_CFL', 'sylvania_CFL', 'ge_incandescent', 'philips_incandescent']

    # #I think you need to go back to filtering - losing information in signal?
    # # compare_different_sensitivity_brfs(ecosmart_CFL_14w, ecosmart_CFL_14w)
    # window_size = 61
    # avg_window = 10
    # pcoeff_list = []
    # pcoeff_name_list = []
    # peak_cross_corr_list = []
    # cross_corr_name_list = []

    # pcoeff_heatmap = []
    # cross_corr_heatmap = []
    
    # ecosmart_avg_1, ecosmart_avg_2 = return_average_periods(ecosmart_CFL_14w, path_3, window_size, avg_window)
    # maxlite_avg_1,maxlite_avg_2 = return_average_periods(maxlite_CFL_15w, path_3,  window_size, avg_window)
    # sylvania_avg_1, sylvania_avg_2 = return_average_periods(sylvania_CFL_13w, path_3, window_size, avg_window)
    # ge_avg_1, ge_avg_2 = return_average_periods(ge_incandescant_25w, path_2, window_size, avg_window)
    # philips_avg_1, philips_avg_2 = return_average_periods(philips_incandescent_40w, path_2, window_size, avg_window)

    # brf_list_1 = [ecosmart_avg_1, maxlite_avg_1, sylvania_avg_1, ge_avg_1, philips_avg_1]
    # brf_list_1.reverse() #reversing list_1 to have heatmap going from bottom left to top right
    # brf_list_2 = [ecosmart_avg_2, maxlite_avg_2, sylvania_avg_2, ge_avg_2, philips_avg_2]

    # for brf_1 in brf_list_1:
    #     for brf_2 in brf_list_2:
    #         # pcoeff_list.append(round(pearson_coeff(brf_1, brf_2), 3))
    #         peak_cross_corr_list.append(round(compare_averaged_brfs(brf_1, brf_2), 3))
    #     # pcoeff_heatmap.append(pcoeff_list)
    #     cross_corr_heatmap.append(peak_cross_corr_list)
    #     # pcoeff_list = []
    #     peak_cross_corr_list = []
    
    # fig, ax = plt.subplots()
    # # pcoeff_heatplot = ax.imshow(pcoeff_heatmap, cmap = "Blues")
    # cross_corr_heatplot = ax.imshow(cross_corr_heatmap, cmap = "Blues")
    # ax.set_xticks(np.arange(len(brf_list_1)))
    # ax.set_yticks(np.arange(len(brf_list_2)))
    # ax.set_xticklabels(brf_name_list_2)
    # ax.set_yticklabels(brf_name_list_1)
    # # ax.set_title("Pearson Correlation Heat Map")
    # ax.set_title('Averaged Cycle Cross Correlation Heat Map')

    # for i in range(len(pcoeff_heatmap)):
    #     for j in range(len(pcoeff_heatmap[i])):
    #         text = ax.text(j, i, pcoeff_heatmap[i][j],
    #                     ha="center", va="center", color="black")

    # for i in range(len(cross_corr_heatmap)):
    #     for j in range(len(cross_corr_heatmap[i])):
    #         text = ax.text(j, i, cross_corr_heatmap[i][j],
    #                     ha="center", va="center", color="black")

    # colorbar = fig.colorbar(cross_corr_heatplot)
    # plt.xticks(rotation = 45)
    # fig.tight_layout(pad = 4.0)
    # plt.show()

    # fig = plt.figure()
    # colorbar = fig.colorbar(plt)
    # plt.imshow(pcoeff_heatmap, cmap = 'Greens')
    # ticks = np.arange(len(brf_list_1))
    # plt.xticks(ticks, brf_name_list, rotation = 45)
    # plt.yticks(ticks, brf_name_list)
    # plt.tight_layout()
    # plt.show()