import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import stats
from scipy.fftpack import fft, ifft
import math

eiko_cfl_13w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_cfl_13w'
eiko_cfl_23w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_cfl_23w'
philips_cfl_13w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\philips_cfl_13w'
eiko_incan_100w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_incandescent_100w'
halco_incan_60w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\halco_incandescent_60w'
philips_incan_200w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\philips_incandescent_200w'

eiko_cfl_13w_0_path = eiko_cfl_13w + '\\' + 'waveform_0.csv' #figure out some automated way for this
eiko_cfl_13w_name = 'Eiko CFL 13W'

eiko_cfl_13w_9_path = eiko_cfl_13w + '\\' + 'waveform_9.csv'

eiko_cfl_23w_0_path = eiko_cfl_23w + '\\' + 'waveform_0.csv'
eiko_cfl_23w_name = 'Eiko CFL 23W'

philips_cfl_13w_path = philips_cfl_13w + '\\' + 'waveform_0.csv'
philips_cfl_13w_name = 'Philips CFL 13W'

eiko_incan_100w_path = eiko_incan_100w + '\\' + 'waveform_0.csv'
eiko_incan_100w_name = 'Eiko Incandescent 100W'

halco_incan_60w_path = halco_incan_60w + '\\' + 'waveform_0.csv'
halco_incan_60w_name = 'Halco Incandescent 60W'

philips_incan_200w_path = philips_incan_200w + '\\' + 'waveform_0.csv'
philips_incan_200w_name = 'Philips Incandescent 200W'

#is there a fast/efficient way to load all the data so that I can just pick and choose without having to
#hardcode path in?

savgol_window = 31

class scope_brf():
    def __init__(self, csv_path):
        brf_data = np.genfromtxt(csv_path, delimiter = ',')
        brf_data = brf_data[1:len(brf_data)] #removing first row
        time, brf, voltage = np.hsplit(brf_data, len(brf_data[0]))
        self.time = np.hstack(time)
        self.brf = np.hstack(brf)
        self.voltage = np.hstack(voltage)

def show_plot(brf_data):
    plt.plot(brf_data)
    plt.show()

def savgol(brf):
    smoothed = ss.savgol_filter(brf, savgol_window, 3)
    return smoothed

def extract_brf(scope_data):
    brf = scope_data.brf
    smoothed = savgol(brf)
    return smoothed

def moving_average(brf, window):
    averaged = np.zeros(len(brf)-window)
    for i in range(len(brf)-window):
        averaged[i] = np.sum(brf[i:i+window])/window
    show_plot(averaged)

def normalize_brf(brf):
    min = np.amin(brf)
    max = np.amax(brf)
    normalized_brf = (brf[:]-min)/(max-min)
    return normalized_brf

def center_zero(brf, zero_value):
    min = np.amin(brf)
    max = np.amax(brf)
    print(zero_value)
    normalized_brf = (brf[:]-brf[zero_value])/(max-min)
    return normalized_brf

def extract_waveforms(base_path):
    waveform_list = []
    for i in range(10):
        brf_path = base_path + '\\waveform_' + str(i) + '.csv'
        brf = scope_brf(brf_path).brf
        nadir_indices = ss.find_peaks(-brf, distance = 750)[0]
        truncated_brf = brf[nadir_indices[0]:nadir_indices[2]] #first and third nadir
        waveform_list.append(truncated_brf)
    return waveform_list

def concatenate_waveforms(waveform_list):
    waveform = np.array([])
    for i in range(len(waveform_list)):
        waveform = np.concatenate((waveform, waveform_list[i]),0)
    return waveform

def crest_factor(brf):
    peak_value = np.amax(brf)
    rms = math.sqrt(np.sum(np.array(brf))/len(brf))
    crest_factor = peak_value/rms
    return crest_factor

def kurtosis(brf):
    return stats.kurtosis(brf)

#smooth brf, then normalize
def crest_factor_analysis(base_path, brf_list_name):
    waveform_list = extract_waveforms(base_path)
    concatenated = np.array([])
    print(brf_list_name)
    for waveform in waveform_list:
        #I think this is just 1/RMS after normalization
        # print(crest_factor(normalize_brf(savgol(waveform))))
        concatenated = np.concatenate((concatenated,waveform),0)
    # print(crest_factor(normalize_brf(concatenated)))
    print(kurtosis(normalize_brf(concatenated)))
    print()

#smooth, normalize
def process_waveform_0(brf, name): #smoothing does effect dft transformation
    # dft(brf)
    filtered_brf = filter_120hz(brf,name)
    filtered_smoothed = savgol(filtered_brf)
    filtered_normalized = normalize_brf(filtered_smoothed)
    return filtered_normalized

def align_brf_sin(brf, sin, name):
    brf = normalize_brf(brf)
    sin = normalize_brf(sin)
    nadir_indices = ss.find_peaks(-brf, distance = 750)[0]
    truncated_brf = brf[nadir_indices[0]:nadir_indices[2]] #first and third nadir
    nadir_indices = ss.find_peaks(-truncated_brf, distance = 750)[0]
    truncated_sin = sin[:len(truncated_brf)]
    aligned_half_cycle = center_zero(truncated_sin, nadir_indices[1])
    aligned_half_cycle[nadir_indices[1]:] = -aligned_half_cycle[nadir_indices[1]:]
    # aligned_half_cycle = center_zero(truncated_sin, nadir_indices[1])
    smoothed_brf = savgol(truncated_brf)
    plt.plot(smoothed_brf, label = 'BRF')
    plt.plot(aligned_half_cycle, label = 'Grid Voltage')
    plt.title(name)
    plt.legend()
    plt.show()
    subtracted = np.subtract(smoothed_brf,aligned_half_cycle)
    plt.plot(subtracted)
    plt.title(name + ' (Subtraction Performed)')
    plt.show()

def map_in_to_out(waveform, in_min, in_max, out_min, out_max):
    new_brf = []
    brf = np.array(waveform)
    for val in brf:
        mapped = (val - in_min)*(out_max - out_min)/(in_max - in_min) + out_min
        new_brf.append(mapped)
    return new_brf

def cross_corr(brf_1, brf_2):
    # correlate = ss.correlate(brf_1, brf_2[:int(len(brf_2)/2)], mode = 'full')/len(brf_1)
    correlate = np.correlate(brf_1, brf_2, mode = 'full')/len(brf_1)
    correlate = map_in_to_out(correlate, -0.5, 0.5, 0, 1) #maps to 0 and 1
    show_plot(correlate)
    max = round(np.amax(np.array(correlate)), 6)
    return max

def filter_120hz(brf, name):
    sample_rate = 990 * 120
    # sos = ss.butter(120, 200, 'hp', fs=sample_rate, output = 'sos')
    # sos = ss.butter(5, [120, 300, 420, 600, 720], 'bs', fs=sample_rate, output = 'sos')
    sos = ss.butter(7, [115, 125], 'bs', fs = sample_rate, output = 'sos')
    filtered_brf = ss.sosfilt(sos, brf)
    # sos = ss.butter(7, [295, 305], 'bs', fs = sample_rate, output = 'sos')
    # filtered_brf = ss.sosfilt(sos, filtered_brf)
    # sos = ss.butter(7, [415, 425], 'bs', fs = sample_rate, output = 'sos')
    # filtered_brf = ss.sosfilt(sos, filtered_brf)
    # sos = ss.butter(7, [595, 605], 'bs', fs = sample_rate, output = 'sos')
    # filtered_brf = ss.sosfilt(sos, filtered_brf)
    # sos = ss.butter(7, [715, 725], 'bs', fs = sample_rate, output = 'sos')
    # filtered_brf = ss.sosfilt(sos, filtered_brf)

    plt.plot(filtered_brf, label = "120Hz \"Filtered\" BRF")
    plt.plot(brf, label = "Original BRF")
    plt.legend()
    plt.title(name)
    plt.show()
    return filtered_brf

if __name__ == "__main__":
    window = 50
    eiko_cfl_13w_0 = scope_brf(eiko_cfl_13w_0_path) #waveform_0
    eiko_cfl_13w_9 = scope_brf(eiko_cfl_13w_9_path)
    eiko_cfl_23w_0 = scope_brf(eiko_cfl_23w_0_path)
    philips_cfl_13w_0 = scope_brf(philips_cfl_13w_path)
    eiko_incan_100w_0 = scope_brf(eiko_incan_100w_path)
    halco_incan_60w_0 = scope_brf(halco_incan_60w_path)
    philips_incan_200w_0 = scope_brf(philips_incan_200w_path)

    crest_factor_analysis(eiko_cfl_13w, eiko_cfl_13w_name)
    crest_factor_analysis(eiko_cfl_23w, eiko_cfl_23w_name)
    crest_factor_analysis(philips_cfl_13w, philips_cfl_13w_name)
    crest_factor_analysis(eiko_incan_100w, eiko_incan_100w_name)
    crest_factor_analysis(halco_incan_60w, halco_incan_60w_name)
    crest_factor_analysis(philips_incan_200w, philips_incan_200w_name)