import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import math
import os

database_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\bulb_database_master.csv'
base_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files'
savgol_window = 31

class plots():
    def show_plot(waveform):
        plt.plot(waveform)
        plt.show()

class brf_database():
    def __init__(self, database_path):
        whole_database = pd.read_csv(database_path)
        self.brf_database = whole_database.loc[:,['Folder_Name', 'Name', 'Bulb_Type']]

    def return_bulb_types(brf_database):
        return brf_database.Bulb_Type.unique()

    def display_types(brf_database):
        print(brf_database.Bulb_Type.unique())

    def return_bulb_type_waveforms(brf_database, bulb_type):
        return brf_database.loc[brf_database['Bulb_Type'] == bulb_type]

    #ordered by: folder name, name, bulb_type; bulb_type info redundent?
    def database_to_list(database):
        return database.values.tolist()

#extracts time, brf, and voltage data from a CSV file
#extracts cycles
class waveform_data():
    def __init__(self, brf_path):
        brf_data = np.genfromtxt(brf_path, delimiter = ',')
        brf_data = brf_data[1:len(brf_data)] #removing first row
        print(brf_data)
        time, brf, voltage = np.hsplit(brf_data, len(brf_data[0]))
        self.time = np.hstack(time)
        self.brf = np.hstack(brf)
        self.voltage = np.hstack(voltage)

    def clean_brf(brf):
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]
        cleaned_brf = brf[nadir_indices[0]:nadir_indices[2]] #first and third nadir
        return cleaned_brf

    def normalize_brf(brf):
        min = np.amin(brf)
        max = np.amax(brf)
        normalized_brf = (brf[:]-min)/(max-min)
        return normalized_brf

    def savgol(brf, savgol_window):
        smoothed = ss.savgol_filter(brf, savgol_window, 3)
        return smoothed

class brf_extraction():
        def __init__(self, brf_path):
            cwd = os.getcwd()
            brf_list = []
            path = base_path + '\\' + brf_path
            os.chdir(path)
            num_files = (len([name for name in os.listdir(path) if os.path.isfile(name)]))
            os.chdir(cwd)
            print(num_files)
            for i in range(num_files):
                brf_path = path + '\\waveform_' + str(i) + '.csv'
                brf = waveform_data(brf_path).brf
                brf = waveform_data.clean_brf(brf)
                brf_list.append(brf)
            self.brf_list = brf_list

        def concatenate_waveforms(waveform_list):
            waveform = np.array([])
            for i in range(len(waveform_list)):
                waveform = np.concatenate((waveform, waveform_list[i]),0)
            return waveform

class brf_analysis():
    def crest_factor(brf):
        peak_value = np.amax(brf)
        rms = math.sqrt(np.sum(np.array(brf))/len(brf))
        crest_factor = peak_value/rms
        return crest_factor

if __name__ == "__main__":
    database = brf_database(database_path).brf_database
    bulb_types = brf_database.return_bulb_types(database)
    for i in range(len(bulb_types)):
        print(bulb_types[i])
        new_database = brf_database.return_bulb_type_waveforms(database,str(bulb_types[i]))
        same_type_list = brf_database.database_to_list(new_database)
        for item in same_type_list:
            print(item[0])
            brf_list = brf_extraction(item[0]).brf_list
            concatenated_brf = brf_extraction.concatenate_waveforms(brf_list)
            plots.show_plot(concatenated_brf)