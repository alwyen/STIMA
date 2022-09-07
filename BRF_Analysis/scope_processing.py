import numpy as np
from scipy import signal
import os
import glob
import define

#extracts time, brf, and voltage data from a CSV file
#does initial processing, such as cleaning the raw data (extracting clean cycles), normalizes, and smooths
class raw_waveform_processing():
    def __init__(self, brf_path):
        brf_data = np.genfromtxt(brf_path, delimiter = ',')
        brf_data = brf_data[1:len(brf_data)] #removing first row
        time, brf, voltage = np.hsplit(brf_data, len(brf_data[0]))
        self.time = np.hstack(time)
        self.brf = np.hstack(brf)
        self.voltage = np.hstack(voltage)
        # brf_df = pd.read_csv(brf_path)
        # time = np.array((brf_df['Time']))
        # brf = np.array((brf_df['Intensity']))
        # voltage = np.array((brf_df['Voltage']))
        # self.time = time
        # self.brf = brf
        # self.voltage = voltage

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
#NOTE: 750 IS HARDCODED; NEEDS TO BE CHANGED/AUTOMATED
# SOLUTION: find max values of BRF and get distance from that
    def clean_brf(time, brf):
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]
        corresponding_time = time[nadir_indices[0]:nadir_indices[2]]
        cleaned_brf = brf[nadir_indices[0]:nadir_indices[2]] #first and third nadir
        return corresponding_time, cleaned_brf
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

    #normalize an array to be between 0 and 1
    def normalize(array):
        min = np.amin(array)
        max = np.amax(array)
        normalized = (array-min)/(max-min)
        return normalized

    #savgol filter
    def savgol(brf, savgol_window):
        smoothed = signal.savgol_filter(brf, savgol_window, 3)
        return smoothed

    #moving average; just done with convolution of a np.ones array
    def moving_average(data, window_size):
        return signal.convolve(data, np.ones(window_size) , mode = 'valid') / window_size

    #truncates the longer BRF
    def truncate_longer(brf_1, brf_2):
        if len(brf_1) > len(brf_2):
            brf_1 = brf_1[0:len(brf_2)]
        else:
            brf_2 = brf_2[0:len(brf_1)]
        return brf_1, brf_2

#uses the raw_waveform_processing class to extract the processed data
#brf_extraction class manipulates data into a list of BRF waveforms or one concatenated BRF waveform
class brf_extraction():
    def __init__(self, brf_folder_name, single_or_double):
        cwd = os.getcwd()
        time_list = []
        brf_list = []
        path = define.base_path + os.sep + brf_folder_name
        os.chdir(path)
        csv_list = glob.glob('*.csv')
        # num_files = (len([name for name in os.listdir(path) if os.path.isfile(name)]))
        # assert len(csv_list) == num_files
        for brf_path in csv_list:
            # brf_path = path + os.sep + 'waveform_' + str(i) + '.csv'
            processed = raw_waveform_processing(brf_path)
            time = processed.time
            brf = processed.brf
            time, brf = raw_waveform_processing.clean_brf(time, brf)
            brf = raw_waveform_processing.normalize(brf)
            nadir_indices = signal.find_peaks(-brf, distance = 750)[0]
            time1, brf1 = time[0:nadir_indices[1]], brf[0:nadir_indices[1]]
            time2, brf2 = time[nadir_indices[1]:len(brf)], brf[nadir_indices[1]:len(brf)]
            if single_or_double == 'double':
                time_list.append(time)
                brf_list.append(brf)
            elif single_or_double == 'single':
                time_list.append(time1)
                time_list.append(time2)
                brf_list.append(brf1)
                brf_list.append(brf2)
        self.time_list = time_list
        self.brf_list = brf_list
        os.chdir(cwd)

    def concatenate_waveforms(waveform_list):
        waveform = np.array([])
        for i in range(len(waveform_list)):
            waveform = np.concatenate((waveform, waveform_list[i]),0)
        return waveform