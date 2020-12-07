import time
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.neighbors import KNeighborsClassifier
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

#this class does all the processing on the database side from the master CSV file
class database_processing():
    '''
    Folders to drop:
        halco_led_10w
        philips_led_8w
        philips_led_6p5w
        sylvania_led_7p5w
        sylvania_led_12w
        westinghouse_led_5p5w
        westinghouse_led_11w

    Interesting BRFs:
        sylvania_led_11w
    '''

    def __init__(self, database_path):
        whole_database = pd.read_csv(database_path)
        self.brf_database = whole_database.loc[:,['Folder_Name', 'Name', 'Bulb_Type']]

    def return_bulb_types(brf_database):
        return brf_database.Bulb_Type.unique()

    def display_types(brf_database):
        print(brf_database.Bulb_Type.unique())

    def return_bulb_type_waveforms(brf_database, bulb_type):
        return brf_database.loc[brf_database['Bulb_Type'] == bulb_type]

    def drop_bulb_type_column(brf_database):
        return brf_database.drop('Bulb_Type', axis = 1)

    #ordered by: folder name, name, bulb_type; bulb_type info redundent?
    def database_to_list(brf_database):
        return brf_database.values.tolist()

#extracts time, brf, and voltage data from a CSV file
#does initial processing, such as cleaning the raw data (extracting clean cycles), normalizes, and smooths
class raw_waveform_processing():
    #
    def __init__(self, brf_path):
        brf_data = np.genfromtxt(brf_path, delimiter = ',')
        brf_data = brf_data[1:len(brf_data)] #removing first row
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

#uses the raw_waveform_processing class to extract the processed data
#brf_extraction class manipulates data into a list of BRF waveforms or one concatenated BRF waveform
class brf_extraction():
        def __init__(self, brf_folder_name):
            cwd = os.getcwd()
            brf_list = []
            path = base_path + '\\' + brf_folder_name
            os.chdir(path)
            num_files = (len([name for name in os.listdir(path) if os.path.isfile(name)]))
            os.chdir(cwd)
            for i in range(num_files):
                brf_path = path + '\\waveform_' + str(i) + '.csv'
                brf = raw_waveform_processing(brf_path).brf
                brf = raw_waveform_processing.normalize_brf(raw_waveform_processing.clean_brf(brf))
                brf_list.append(brf)
            self.brf_list = brf_list

        def concatenate_waveforms(waveform_list):
            waveform = np.array([])
            for i in range(len(waveform_list)):
                waveform = np.concatenate((waveform, waveform_list[i]),0)
            return waveform

#brf_analysis class contains all the statistical tests/analysis methods
class brf_analysis():
    def crest_factor(brf):
        peak_value = np.amax(brf)
        rms = math.sqrt(np.sum(np.array(brf))/len(brf))
        crest_factor = peak_value/rms
        return crest_factor

    def kurtosis(brf):
        return stats.kurtosis(brf)

    def skew(brf):
        return stats.skew(brf)

    #for each bulb type, print out the stats for that particular concatenated waveform
    def for_type_print_stats(brf_database):
        bulb_types = database_processing.return_bulb_types(brf_database) #drop bulb type column later?
        for i in range(len(bulb_types)):
            print(bulb_types[i])
            new_database = database_processing.return_bulb_type_waveforms(brf_database,str(bulb_types[i]))
            same_type_list = database_processing.database_to_list(new_database)
            for item in same_type_list: #item: folder name; name; bulb type
                brf_list = brf_extraction(item[0]).brf_list
                concatenated_brf = brf_extraction.concatenate_waveforms(brf_list)
                # print(brf_analysis.crest_factor(concatenated_brf))
                # print(brf_analysis.kurtosis(concatenated_brf))
                print(brf_analysis.skew(concatenated_brf))
            print()

class brf_classification():
    #split this up into multiple functions?
    def KNN(brf_database):
        # brf_database = database_processing.drop_bulb_type_column(brf_database)
        # database_as_list = database_processing.database_to_list(brf_database)

        bulb_type_list = database_processing.return_bulb_types(brf_database)

        for bulb_type in bulb_type_list:
            print(bulb_type)
            same_type_database = database_processing.return_bulb_type_waveforms(brf_database,str(bulb_type))
            same_type_database = database_processing.drop_bulb_type_column(same_type_database)
            same_type_list = database_processing.database_to_list(same_type_database)

            KNN_input = list([])
            KNN_output = list([])

            #for each element:
            #   index 0: [crest factor, kurtosis, skew]
            #   index 1: BRF name
            KNN_prediction_list = list([])

            #index 0: Folder Name
            #index 1: BRF Name
            for i in range(len(same_type_list)):
                folder_name = same_type_list[i][0]
                brf_name = same_type_list[i][1]
                waveform_list = brf_extraction(folder_name).brf_list
                #ignoring the first waveform – will use that for classification
                for i in range(len(waveform_list)):
                    crest_factor = brf_analysis.crest_factor(waveform_list[i])
                    kurtosis = brf_analysis.kurtosis(waveform_list[i])
                    skew = brf_analysis.skew(waveform_list[i])
                    input_param = [crest_factor, kurtosis, skew]
                    if i == 0:
                        KNN_prediction_list.append([input_param, brf_name])
                    else:
                        KNN_input.append(input_param)
                        KNN_output.append(brf_name)
                print(f'{brf_name} Finished')

            brf_KNN_model = KNeighborsClassifier(n_neighbors = 9) #used 9 because about 9 waveforms for each BRF
            brf_KNN_model.fit(KNN_input, KNN_output)
            # print('Model Built')
            
            true_positive = 0
            total = len(KNN_prediction_list)
            for prediction in KNN_prediction_list:
                input_data = prediction[0]
                # print(input_data)
                expected_output = prediction[1]
                output = brf_KNN_model.predict([input_data])[0]
                if expected_output == output:
                    true_positive += 1

            recall = true_positive/total
            print(f'Recall: {recall}')
            print()

if __name__ == "__main__":
    brf_database = database_processing(database_path).brf_database

    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'halco_led_10w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_6p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_8w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_7p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_12w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_5p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_11w'].index)

    brf_classification.KNN(brf_database)