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

path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\BRF_images'

#make classes man..

ecosmart_CFL_14w = 'ecosmart_CFL_14w'
maxlite_CFL_15w = 'maxlite_CFL_15w'
ge_incandescant_25w = 'ge_incandescant_25w'
philips_incandescent_40w = 'philips_incandescent_40w'

height = 576
width = 1024

def get_rolling_dc(bulb):
    img_paths = path + '\\' + bulb
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

def fit_curve(brf):
    x = np.arange(0, len(brf), 1)
    brf_model = interpolate.interp1d(x, brf, kind = 'cubic')
    interpolated_brf = brf_model(x)
    plt.plot(interpolated_brf)
    plt.show()

def upper_envelope(brf):
    peak_indices = ss.find_peaks(brf)[0]
    peak_values = brf[peak_indices]
    # upper_model = interpolate.interp1d(peak_indices, peak_values, fill_value='extrapolate')
    upper_model = interpolate.UnivariateSpline(peak_indices, peak_values)
    upper_envelope_values = np.zeros(len(brf))
    upper_envelope_values[peak_indices] = peak_values
    upper_envelope = upper_model(np.arange(0, len(brf), 1))

    return upper_envelope

def lower_envelope(brf):
    trough_indices = ss.find_peaks(-brf)[0]
    trough_values = brf[trough_indices]
    # lower_model = interpolate.interp1d(trough_indices, trough_values, fill_value='extrapolate')
    lower_model = interpolate.UnivariateSpline(trough_indices, trough_values)
    lower_envelope_values = np.zeros(len(brf))
    lower_envelope_values[trough_indices] = trough_values
    lower_envelope = lower_model(np.arange(0, len(brf), 1))

    return lower_envelope

def upper_lower_envelope_mean(upper_envelope, lower_envelope):
    processed_envelope = (upper_envelope + lower_envelope)/2
    return processed_envelope

def smooth_interpolation(brf):
    x = np.arange(0, len(brf), 1)
    univ_spline = interpolate.UnivariateSpline(x, brf)

    plt.plot(x, univ_spline(x))
    plt.show()

def moving_average(image_column, window_size):
    average = []
    for x in range(len(image_column)-window_size):
        average.append(np.sum(image_column[x:x+window_size])/window_size/255)
    return average

#maybe just get rid of this..
def savitzky_golay_filter(brf, window_length, polyorder):
    filtered_data = ss.savgol_filter(brf, window_length, polyorder) #array, window_length, order of polynomial to fit samples; win > poly
    return filtered_data

def crop_brf(brf, start_index, end_index):
    return brf[start_index:end_index]

def normalize_img(rolling_img, dc_img):
    normalized_img = np.divide(rolling_img, dc_img)
    height = normalized_img.shape[0]
    width = normalized_img.shape[1]
    normalized_img = normalized_img/(np.sum(normalized_img)/width/height)
    return normalized_img

def align_brfs(brf_1, bulb_1_name, brf_2, bulb_2_name):
    #shfit amount corresponds to second argument (brf_2)
    shift_amount = phase_align(brf_1, brf_2, [10, 90]) #[10, 90] => region of interest; figure out what this is???
    shifted_brf_2 = shift(brf_2, shift_amount, mode = 'nearest')
    # plt.figure(figsize = (10, 2))
    # plt.plot(brf_1, label = bulb_1_name)
    # plt.plot(shifted_brf_2, label = bulb_2_name)
    # plt.legend(loc = 'best')
    # plt.show()
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

def dtw_method(brf_1, brf_2):
    array = np.array([brf_1, brf_2])
    array = array.reshape(len(brf_1), len(array))
    df = pd.DataFrame(array)
    s1 = df.iloc[:,0].interpolate().values
    s2 = df.iloc[:,1].interpolate().values
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(s1, s2, dist = 'euclidean')
    print(np.round(d,2))

def compare_brfs(bulb_1, bulb_2, savgov_window):
    img_rolling_1, img_dc_1 = get_rolling_dc(bulb_1)
    img_rolling_1 = img_from_path(img_rolling_1)
    img_dc_1 = img_from_path(img_dc_1)
    normalized_1 = normalize_img(img_rolling_1, img_dc_1)

    img_rolling_2, img_dc_2 = get_rolling_dc(bulb_2)
    img_rolling_2 = img_from_path(img_rolling_2)
    img_dc_2 = img_from_path(img_dc_2)
    normalized_2 = normalize_img(img_rolling_2, img_dc_2)

    brf_rolling_1 = brf_extraction(normalized_1)
    brf_rolling_2 = brf_extraction(normalized_2)

    # brf_rolling_1 = crop_brf(brf_extraction(normalized_1), 0, 500)
    # brf_rolling_2 = crop_brf(brf_extraction(normalized_2), 0, 500)

    # smoothed_brf_1 = savitzky_golay_filter(brf_rolling_1, savgov_window, 3)
    # smoothed_brf_2 = savitzky_golay_filter(brf_rolling_2, savgov_window, 3)
    smoothed_brf_1 = brf_rolling_1
    smoothed_brf_2 = brf_rolling_2

    aligned_brf_1, aligned_brf_2 = align_brfs(smoothed_brf_1, bulb_1, smoothed_brf_2, bulb_2)
    plt.plot(aligned_brf_1)
    plt.plot(aligned_brf_2)
    plt.show()

    cropped_brf_1 = crop_brf(smoothed_brf_1, 0, 250)

    correlate_1 = ss.correlate(cropped_brf_1, aligned_brf_2, mode = 'full')/len(cropped_brf_1)
    correlate_1 = crop_brf(correlate_1, len(cropped_brf_1), len(correlate_1) - len(cropped_brf_1))
    return correlate_1

    # correlate_2 = ss.correlate(cropped_brf_1, aligned_brf_1, mode = 'full')/len(cropped_brf_1)
    # correlate_2 = crop_brf(correlate_2, len(cropped_brf_1), len(correlate_1) - len(cropped_brf_1))

    # correlate_3 = ss.correlate(cropped_brf_1, smoothed_brf_2, mode = 'full')/len(cropped_brf_1)
    # correlate_3 = crop_brf(correlate_3, len(cropped_brf_1), len(correlate_1) - len(cropped_brf_1))

if __name__ == '__main__':
    #I think you need to go back to filtering - losing information in signal?
    for window_size in range(61, 70, 2):
        print(f'Window Size: {window_size}')
        cfl_cfl_self = compare_brfs(ecosmart_CFL_14w, ecosmart_CFL_14w, window_size)
        cfl_cfl = compare_brfs(ecosmart_CFL_14w, maxlite_CFL_15w, window_size)
        inc_inc = compare_brfs(ge_incandescant_25w, philips_incandescent_40w, window_size)
        cfl_inc = compare_brfs(ecosmart_CFL_14w, philips_incandescent_40w, window_size)

        plt.plot(cfl_cfl_self, label = 'self')
        plt.plot(cfl_cfl, label = 'cfl_cfl')
        plt.plot(inc_inc, label = 'inc_inc')
        plt.plot(cfl_inc, label = 'cfl_inc')
        plt.legend()
        plt.show()

    # img = img_from_path(img_path)
    # brf = brf_extraction(img)
    # brf = np.array(brf) #conversion into numpy array
    # # brf = crop_brf(brf, 0, 250)
    # brf = crop_brf(brf, 0, 500)
    # # brf = moving_average(brf, 5)
    # # show_plot(brf)

    # u_e = upper_envelope(brf)
    # l_e = lower_envelope(brf)
    # processed_brf = upper_lower_envelope_mean(u_e, l_e)
    # upper_processed = upper_envelope(processed_brf)
    # lower_processed = lower_envelope(processed_brf)
    # processed_average = moving_average(processed_brf, 10)
    # brf_average = moving_average(brf, 5)


    # print('BRF')
    # show_plot(brf)

    # print('Processed')
    # show_plot(processed_brf)

    # print('Savgov filter')
    # for x in range(23, 62, 2):
    #     print(x)
    #     savitzky_golay_filter(brf, x, 3)
        # savitzky_golay_filter(brf_average, x, 3)

    # print('Moving Average')
    # brf_average = moving_average(processed_brf, 5)
    # show_plot(brf_average)

    # brf = envelope_processing(brf)
    # envelope_processing(brf_average)
    # show_plot(brf)