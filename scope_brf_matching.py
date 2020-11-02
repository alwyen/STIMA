import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.fftpack import fft, ifft
eiko_cfl_13w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_cfl_13w'
eiko_cfl_23w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_cfl_23w'
philips_cfl_13w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\philips_cfl_13w'
eiko_incandescent_100w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_incandescent_100w'

eiko_cfl_13w_0_path = eiko_cfl_13w + '\\' + 'waveform_0.csv' #figure out some automated way for this
eiko_cfl_13w_9_path = eiko_cfl_13w + '\\' + 'waveform_9.csv'
eiko_cfl_23w_0_path = eiko_cfl_23w + '\\' + 'waveform_0.csv'
philips_cfl_13w_path = philips_cfl_13w + '\\' + 'waveform_0.csv'
eiko_incan_100w_path = eiko_incandescent_100w + '\\' + 'waveform_0.csv'
#is there a fast/efficient way to load all the data so that I can just pick and choose without having to
#hardcode path in?

savgol_window = 31

class scope_brf():
    def __init__(self, csv_path):
        brf_data = np.genfromtxt(csv_path, delimiter = ',')
        brf_data = brf_data[1:len(brf_data)] #removing first row
        time, brf, voltage = np.hsplit(brf_data, len(brf_data[0]))
        self.time = time
        self.brf = brf
        self.voltage = voltage

def show_plot(brf_data):
    plt.plot(brf_data)
    plt.show()

def savgol(brf):
    smoothed = ss.savgol_filter(brf, savgol_window, 3)
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

#hstack, smooth, normalize
def process_brf_0(brf): #smoothing does effect dft transformation
    brf = np.hstack(brf) #combining into 1D array from len(brf) array
    # dft(brf)
    smoothed = savgol(brf)
    normalized = normalize_brf(smoothed)
    dft(normalized)
    # peak_indices = ss.find_peaks(-normalized, distance = 800)[0]
    # peak_values = normalized[peak_indices]
    # plt.plot(peak_indices, peak_values, 'x')
    # plt.plot(normalized)
    # plt.show()
    # for i in range(len(peak_indices)-1):
        # print(peak_indices[i+1]-peak_indices[i])

    # show_plot(normalized)
    return normalized

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

def dft(brf):
    boundary = 50
    
    sample_rate = 990 * 120 #hardcoded sample rate: 
    num_samples = len(brf)
    # print(num_samples)
    sample_spacing = 1/sample_rate
    # print(sample_spacing)
    xf = np.linspace(0, 1.0/(2*sample_spacing), num_samples//2)
    # print(xf)
    yf_orig = fft(brf)

    yf = 2.0/num_samples*np.abs(yf_orig[0:num_samples//2])
    # yf = normalize_brf(yf)
    peak_indices = ss.find_peaks(yf)[0]
    xf_values = xf[peak_indices]
    print(xf_values[:5])
    yf_temp = yf
    plt.plot(xf[:boundary], yf[:boundary])
    yf_temp[peak_indices[0]] = 0
    print(yf_temp[peak_indices[0]])
    plt.plot(xf[:boundary], yf_temp[:boundary])
    plt.show()

    # plt.plot(np.fft.irfft(yf))
    # # plt.plot(brf)
    # plt.show()

if __name__ == "__main__":
    window = 50
    eiko_cfl_13w_0 = scope_brf(eiko_cfl_13w_0_path) #waveform_0
    eiko_cfl_13w_9 = scope_brf(eiko_cfl_13w_9_path)
    eiko_cfl_23w_0 = scope_brf(eiko_cfl_23w_0_path)
    philips_cfl_13w_0 = scope_brf(philips_cfl_13w_path)
    eiko_incan_100w_0 = scope_brf(eiko_incan_100w_path)

    eiko_cfl_13w_0_brf = process_brf_0(eiko_cfl_13w_0.brf)
    # moving_average(eiko_cfl_13w_0_brf, window)
    eiko_cfl_13w_9_brf = process_brf_0(eiko_cfl_13w_9.brf)
    eiko_cfl_23w_0_brf = process_brf_0(eiko_cfl_23w_0.brf)
    philips_cfl_13w_0_brf = process_brf_0(philips_cfl_13w_0.brf)
    eiko_incan_100w_0_brf = process_brf_0(eiko_incan_100w_0.brf)
    # moving_average(eiko_incan_100w_0_brf, window)

    #eiko cfl 13w 0 with 9
    # cc1 = cross_corr(eiko_cfl_13w_0_brf, eiko_cfl_13w_9_brf)

    #eiko cfl 13w_0 with 23w_0
    # cc2 = cross_corr(eiko_cfl_13w_0_brf, eiko_cfl_23w_0_brf)

    #eiko 13w with philips 13w
    # cc3 = cross_corr(eiko_cfl_13w_0_brf, philips_cfl_13w_0_brf)

    #eiko cfl 13w with philips incan 100w
    # cc4 = cross_corr(eiko_cfl_13w_0_brf, eiko_incan_100w_0_brf)

    # print(cc1)
    # print(cc2)
    # print(cc3)
    # print(cc4)