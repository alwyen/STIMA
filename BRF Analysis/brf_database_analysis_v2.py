import time
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt
import math
import os
import seaborn as sn

import pdb

#CLEAN UP TIME
#TODO: feature_analysis: condense and make it so that I can just include different feature names and they'll spit out the right graphs; clear_lists has gotta go
#fix all the global methods/variable names
#remove data_processing class methods that aren't being used; just look up pandas methods dude
#make export_all_to_csv dynamic; maybe the solution is to hand in a list of features to be analyzed? somehow need to call the method name then; associate "method name" with "name"?
#check is "train_KNN" and "KNN" can be simplified

database_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\bulb_database_master.csv'
base_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files'
gradient_save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Gradient Tests\Savgol 31 Moving 50'
raw_waveform_save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\Raw BRFs'
savgol_window = 31
mov_avg_w_size = 50

brf_analysis_save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\BRF Analysis'
feature_analysis_save_directory = r'C:\Users\alexy\OneDrive\Documents\STIMA\BRF Analysis\Feature Analysis'
############################################################
#debugging
falling_slope = 0
rising_slope = 0
nadir = 0
x1 = None
y1 = None
x2 = None
y2 = None

x_pt1 = None
y_pt1 = None

x_pt2 = None
y_pt2 = None

#these methods are for showing lines on a graph
def set_x1(data):
    global x1
    x1 = data

def set_y1(data):
    global y1
    y1 = data

def set_x2(data):
    global x2
    x2 = data

def set_y2(data):
    global y2
    y2 = data

#set_pt1 and set_pt2 are for displaying a point on a graph
def set_pt1(x, y):
    global x_pt1
    global y_pt1

    x_pt1 = x
    y_pt1 = y

def set_pt2(x, y):
    global x_pt2
    global y_pt2

    x_pt2 = x
    y_pt2 = y

############################################################

#for skipping plotting in KNN
skip = False

def skip_plots(val):
    global skip
    skip = val

#plotting
class plots():
    #show plot
    def show_plot(waveform):
        plt.plot(waveform)
        plt.show()
    
    #save plot
    def save_plot(waveform, plot_name):
        plt.plot(waveform)
        plt.title(plot_name)
        plt.savefig(plot_name + '.png')
        plt.clf()

    #confusion matrix for bulb types only
    def confusion_matrix_type(ground_list, predicted_list, bulb_types, title):
        bulb_types_list = list(bulb_types[:4])
        bulb_types = list(bulb_types)
        prediction_matrix = np.zeros((len(bulb_types_list),len(bulb_types_list)))
        total = np.zeros((len(bulb_types_list),len(bulb_types_list)))

        assert len(ground_list) == len(predicted_list)

        for i in range(len(ground_list)):
            ground_index = bulb_types.index(ground_list[i])
            predicted_index = bulb_types.index(predicted_list[i])
            if ground_index < 4 and predicted_index < 4: #excluding halogen-xenon and halogen-incandescent
                prediction_matrix[ground_index][predicted_index] += 1
                total[ground_index][:] += 1

        confusion_matrix = np.divide(prediction_matrix, total)
        
        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in bulb_types_list], columns = [i for i in bulb_types_list])
        #(12,7) if just figure title
        plt.figure(figsize = (13,9))
        sn.heatmap(df_cm, annot=True)
        plt.title(title)
        plt.xlabel('Predicted', fontsize = 16)
        plt.ylabel('Expected', fontsize = 16)
        plt.yticks(va = 'center')
        plt.show()

    #issues with the ticks for a confusion matrix of the entire BRF database
    #show the confusion matrix for unique BRFs
    def confusion_matrix_unique(ground_list, predicted_list, unique_brf_names, title):
        prediction_matrix = np.zeros((len(unique_brf_names),len(unique_brf_names)))
        total_matrix = np.zeros((len(unique_brf_names),len(unique_brf_names)))

        assert len(ground_list) == len(predicted_list)

        for i in range(len(ground_list)):
            ground_index = unique_brf_names.index(ground_list[i])
            predicted_index = unique_brf_names.index(predicted_list[i])
            prediction_matrix[ground_index][predicted_index] += 1
            total_matrix[ground_index][:] += 1

        confusion_matrix = np.divide(prediction_matrix, total_matrix)

        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in unique_brf_names], columns = [i for i in unique_brf_names])
        plt.figure(figsize = (8,7)).tight_layout()
        sn.heatmap(df_cm)
        plt.title(title)
        plt.xlabel('Predicted', fontsize = 16)
        #font size is too small, and not sure if can make it smaller
        plt.xticks(np.arange(0.5, len(unique_brf_names), 1), unique_brf_names, rotation = 45, ha = 'right', fontsize = 6)
        plt.ylabel('Expected', fontsize = 16)
        plt.yticks(np.arange(0.5, len(unique_brf_names), 1), unique_brf_names, rotation = 45, va = 'top', fontsize = 6)
        plt.tight_layout()
        plt.show()

    def misclass_bar_graph(name_list, misclassification_array, plot_title):
        # plt.figure(figsize = (15,8))
        plt.figure(figsize = (15,5))
        plt.bar(name_list, misclassification_array)
        plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
        plt.tight_layout()
        plt.title(plot_title)
        plt.show()

    def k_and_k_fold(k_list, overall_accuracy, cfl_accuracy, halogen_accuracy, incandescent_accuracy, led_accuracy, title):
        
        plt.plot(overall_accuracy, label = 'Overall Accuracy')
        plt.plot(cfl_accuracy, label = 'CFL Accuracy')
        plt.plot(halogen_accuracy, label = 'Halogen Accuracy')
        plt.plot(incandescent_accuracy, label = 'Incandescent Accuracy')
        plt.plot(led_accuracy, label = 'LED Accuracy')
        plt.xticks(np.arange(0,len(k_list), 1), k_list.astype('str'))
        plt.xlabel('Value of \'k\' for KNN')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
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
    #reads the database path
    def __init__(self, database_path):
        whole_database = pd.read_csv(database_path)
        self.brf_database = whole_database.loc[:,['Folder_Name', 'Name', 'Bulb_Type']]

    #returns dataframe of a specific bulb type
    def same_type_df(brf_database, bulb_type):
        return brf_database.loc[brf_database['Bulb_Type'] == bulb_type]

    #ordered by: [folder name, name, bulb_type]
    #the database refers to the Master CSV file
    def database_to_list(brf_database):
        return brf_database.values.tolist()

    #for every waveform, export each feature (peak_location and integral_average not included) into a single CSV file
    def export_all_to_csv(brf_database, method_name_list, single_or_double):
        brf_database_list = database_processing.database_to_list(brf_database)

        #NOTE: "Integral ratio" AND "angle of inflection" BOTH USED SMOOTHED WAVEFORMS; "crest factor," "kurtosis," and "skew" DO NOT
        name_list = list([])
        type_list = list([])
        integral_ratio_list = list([])
        nadir_angle_list = list([])
        crest_factor_list = list([])
        kurtosis_list = list([])
        skew_factor_list = list([])

        for i in range(len(brf_database_list)):
            smoothed = None
            folder_name = brf_database_list[i][0]
            brf_name = brf_database_list[i][1]
            bulb_type = brf_database_list[i][2]
            extracted_lists = brf_extraction(folder_name, single_or_double)
            time_list = extracted_lists.time_list
            waveform_list = extracted_lists.brf_list

            for j in range(len(waveform_list)):
                smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[j], savgol_window), mov_avg_w_size)
                ratio = brf_analysis.integral_ratio(smoothed, single_or_double)
                nadir_angle = brf_analysis.angle_of_inflection(time_list[j], smoothed, single_or_double, 'nadir')
                crest_factor = brf_analysis.crest_factor(waveform_list[j])
                kurtosis = brf_analysis.kurtosis(waveform_list[j])
                skew_factor = brf_analysis.skew(waveform_list[j])

                name_list.append(brf_name)
                type_list.append(bulb_type)
                integral_ratio_list.append(ratio)
                nadir_angle_list.append(nadir_angle)
                crest_factor_list.append(crest_factor)
                kurtosis_list.append(kurtosis)
                skew_factor_list.append(skew_factor)

            print(brf_name)
        
        # geeksforgeeks.org/create-a-pandas-dataframe-from-lists/ --> USE THIS WAY INSTEAD
        d = {'BRF_Name': name_list, 'Bulb_Type': type_list, 'Integral_Ratio': integral_ratio_list, 'Nadir_Angle': nadir_angle_list, 'Crest_Factor': crest_factor_list, 'Kurtosis': kurtosis_list, 'Skew': skew_factor_list}
        df = pd.DataFrame(data = d)
        save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\BRF Analysis' + '\\stat_analysis.csv'

        if os.path.exists(save_path):
            print('File exists: do you want to overwrite? (Y/N):')
            x = input()
            if x == 'Y':
                df.to_csv(save_path)
        else:
            df.to_csv(save_path)

        return df

    '''
    #pkl file with both the bulb type and the name for output labels
    #ordered by: Input | Bulb_Type | Name

    INPUT:
        brf_database        Entire BRF database
        single_or_double    Single cycle or double cycle?
        num_features        Double check to see if using the right number of features

    OUTPUT:
        pkl file of dataframe that is ordered by:
        
        Features | Bulb_Type | Name
                    ...
    '''
    def pkl_KNN_in_out(brf_database, single_or_double, num_features, save_path):
        brf_database_list = database_processing.database_to_list(brf_database)
        
        KNN_input = list()
        KNN_output_type = list([])
        KNN_output_name = list([])

        row_list = list()

        max_val = -9999
        min_val = 9999
        down_list = list()

        #index 0: Folder Name
        #index 1: BRF Name
        #index 2: Bulb Type
        for i in range(len(brf_database_list)):
            folder_name = brf_database_list[i][0]
            brf_name = brf_database_list[i][1]
            bulb_type = brf_database_list[i][2]
            extracted_lists = brf_extraction(folder_name, single_or_double)
            time_list = extracted_lists.time_list
            waveform_list = extracted_lists.brf_list
            #ignoring the first waveform – will use that for classification
            for i in range(len(waveform_list)):
                downsampled_BRF = raw_waveform_processing.normalize(signal.resample(waveform_list[i], 1600))

                # smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(downsampled_BRF, savgol_window), mov_avg_w_size)
                # # linearity = brf_analysis.linearity(time_list[i], smoothed, single_or_double, 'falling')
                # angle = brf_analysis.angle_of_inflection(time_list[i], smoothed, single_or_double, 'nadir')
                # integral_ratio = brf_analysis.integral_ratio(smoothed, single_or_double)
                # int_avg = brf_analysis.cycle_integral_avg(downsampled_BRF, single_or_double)
                # peak_loc = brf_analysis.peak_location(downsampled_BRF, single_or_double)
                # crest_factor = brf_analysis.crest_factor(downsampled_BRF)
                # kurtosis = brf_analysis.kurtosis(downsampled_BRF)
                # skew = brf_analysis.skew(downsampled_BRF)

                
                smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[i], savgol_window), mov_avg_w_size)
                # linearity = brf_analysis.linearity(time_list[i], smoothed, single_or_double, 'falling')
                # angle = brf_analysis.angle_of_inflection(time_list[i], smoothed, single_or_double, 'nadir')
                integral_ratio = brf_analysis.integral_ratio(smoothed, single_or_double)

                int_avg = brf_analysis.cycle_integral_avg(waveform_list[i], single_or_double)                
                peak_loc = brf_analysis.peak_location(waveform_list[i], single_or_double)
                crest_factor = brf_analysis.crest_factor(waveform_list[i])
                kurtosis = brf_analysis.kurtosis(waveform_list[i])
                skew = brf_analysis.skew(waveform_list[i])

                # int_avg = brf_analysis.cycle_integral_avg(smoothed, single_or_double)                
                # peak_loc = brf_analysis.peak_location(smoothed, single_or_double)
                # crest_factor = brf_analysis.crest_factor(smoothed)
                # kurtosis = brf_analysis.kurtosis(smoothed)
                # skew = brf_analysis.skew(smoothed)
                # input_param = [linearity, angle, integral_ratio, crest_factor, kurtosis, skew]
                # input_param = np.array([crest_factor, kurtosis, skew])
                

                # input_param = np.array([angle, integral_ratio, int_avg, peak_loc, crest_factor, kurtosis, skew])
                input_param = np.array([int_avg, peak_loc, crest_factor, kurtosis, skew])
                
                assert len(input_param) == num_features

                KNN_input.append(input_param)
                KNN_output_type.append(bulb_type)
                KNN_output_name.append(brf_name)

                item_list = list()
                item_list.append(brf_name)
                item_list.append(bulb_type)
                for param in input_param:
                    item_list.append(param)
                for pt in downsampled_BRF:
                    item_list.append(pt)

                row_list.append(item_list)

                downsample_amount = len(waveform_list[i]) - len(downsampled_BRF)
                if downsample_amount < min_val:
                    min_val = downsample_amount
                if downsample_amount > max_val:
                    max_val = downsample_amount
                down_list.append(downsample_amount)


                # plots.show_plot(waveform_list[i])
                # plots.show_plot(downsampled_BRF)

            print(f'{brf_name} Finished') #this is just to make sure the program is running properly

        print(f'Min: {min_val}')
        print(f'Max: {max_val}')
        average = np.sum(np.array(down_list)) / len(down_list)
        print(f'Average: {average}')

        d = {'Features': KNN_input, 'Bulb_Type': KNN_output_type, 'Name': KNN_output_name}
        df = pd.DataFrame(data = d)

        df_zeal = pd.DataFrame(row_list)

        df.to_pickle(save_path + '//KNN_downsized.pkl')
        # df.to_pickle(save_path + '//KNN_downsized_smoothed.pkl')
        # df.to_csv(save_path + '//KNN.csv')
        # df_zeal.to_pickle(save_path + '//Zeal_stats.pkl', protocol = 4)
        # df_zeal.to_csv(save_path + '//Zeal_stats.csv')

    '''
    loads the pkl file and makes a dataframe of the inputs and the outputs ("type" or "name" is specified)

    INPUT:
        pkl_path                pkl file path
        classification_type     "type" or "name" output classification

    OUTPUT:
        KNN_input               list of feature inputs
        KNN_output              feature labels (either "type" or "name")
    '''
    def load_KNN_pkl(pkl_path, classification_type):
        KNN_df = pd.read_pickle(pkl_path)
        KNN_input = KNN_df.Features.tolist()
        if classification_type == 'type':
            KNN_output = KNN_df.Bulb_Type.tolist()
        elif classification_type == 'name':
            KNN_output = KNN_df.Name.tolist()

        return KNN_input, KNN_output


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

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
#NOTE: 750 IS HARDCODED; NEEDS TO BE CHANGED/AUTOMATED
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
            path = base_path + '\\' + brf_folder_name
            os.chdir(path)
            num_files = (len([name for name in os.listdir(path) if os.path.isfile(name)]))
            os.chdir(cwd)
            for i in range(num_files):
                brf_path = path + '\\waveform_' + str(i) + '.csv'
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

        def concatenate_waveforms(waveform_list):
            waveform = np.array([])
            for i in range(len(waveform_list)):
                waveform = np.concatenate((waveform, waveform_list[i]),0)
            return waveform

#brf_analysis class contains all the statistical tests/analysis methods
class brf_analysis():
    #gets distance squared of two BRFs; not euclidean distance (did not sqrt the result)
    #this is the method used in Sheinin's paper
    def min_error(brf_1, brf_2):
        brf_1, brf_2 = raw_waveform_processing.truncate_longer(brf_1, brf_2)
        error = np.sum(np.square(np.absolute(np.array(brf_1) - np.array(brf_2))))
        return error

    #crest factor; just 1/RMS for our case
    def crest_factor(brf):
        peak_value = np.amax(brf)
        rms = math.sqrt(np.sum(np.array(brf))/len(brf))
        crest_factor = peak_value/rms
        return crest_factor

    #kurtosis
    def kurtosis(brf):
        return stats.kurtosis(brf)

    #skew
    def skew(brf):
        return stats.skew(brf)

    #returns gradient of BRF/array
    def gradient(data, name):
        horizontal_gradient = np.array([1, 0, -1])
        # horizontal_gradient = np.array([1/12, -2/3, 0, 2/3, -1/12])
        gradient_x = signal.convolve(data, horizontal_gradient, mode = 'valid')
        smoothed_gradient = raw_waveform_processing.savgol(gradient_x, savgol_window)
        # plots.save_gradient_plot(data, smoothed_gradient, name)
        plots.save_gradient_plot(data, gradient_x, name)
        return gradient_x

    #enforces that the integral (or sum) is equal to 1
    def normalize_cycle(brf):
        integral = np.sum(brf)
        normalized = brf/integral
        return normalized

    #this ratio is defined by the left side of the peak/right side of the peak
    #(rising/falling)
    def integral_ratio(brf, single_or_double):
        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]

        if single_or_double == 'single':
            peak_indice = peak_indices[0]

            rising = brf[0:peak_indice+1]
            falling = brf[peak_indice:len(brf)]

            ratio = np.sum(rising)/np.sum(falling)
            
        elif single_or_double == 'double':
            peak_indice_1 = peak_indices[0]
            peak_indice_2 = peak_indices[1]
            nadir_indice_1 = 0

            #NOTE: temporary solution to picking nadir; maybe better to get approx location of nadir through unsmoothed waveform
            #there was an issue with getting the nadir indice which is why I wrote the following statements
            if nadir_indices[0] < 100:
                nadir_indice_1 = nadir_indices[1]
            else:
                nadir_indice_1 = nadir_indices[0]

            integral_normalized_1 = brf_analysis.normalize_cycle(brf[0:nadir_indice_1])
            integral_normalized_2 = brf_analysis.normalize_cycle(brf[nadir_indice_1+1:len(brf)])

            new_brf = np.hstack((integral_normalized_1, integral_normalized_2))

            rising_1 = new_brf[0:peak_indice_1+1]
            falling_1 = new_brf[peak_indice_1:nadir_indice_1+1]
            rising_2 = new_brf[nadir_indice_1:peak_indice_2+1]
            falling_2 = new_brf[peak_indice_2:len(brf)]

            ratio_1 = np.sum(rising_1)/np.sum(falling_1)
            ratio_2 = np.sum(rising_2)/np.sum(falling_2)

            set_pt1(peak_indice_1, new_brf[peak_indice_1])
            set_pt2(peak_indice_2, new_brf[peak_indice_2])

            average_ratio = (ratio_1 + ratio_2)/2

            return average_ratio

    #not good; method has issues and doesn't seem distinct enough for KNN
    def linearity(time, brf, single_or_double, rising_or_falling):
        nadir_clipping_length = 0
        peak_clipping_length = 0

        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]

        # print(f'Peaks: {peak_indices}')
        # print(f'Nadirs: {nadir_indices}')

        linear_regressor = LinearRegression()

        if single_or_double == 'single':
            peak_indice = peak_indices[0]

            rising = brf[0:peak_indice+1]
            falling = brf[peak_indice:len(brf)]

            rising_line = np.linspace(brf[0], brf[peak_indice], peak_indice+1)
            falling_line = np.linspace(brf[peak_indice], brf[len(brf)-1], len(brf)-peak_indice+1)

            cc_1 = np.corrcoef(rising, rising_line)
            cc_2 = np.corrcoef(falling, falling_line)

            return cc_1[0][1], cc_2[0][1]

        elif single_or_double == 'double':
            peak_indice_1 = peak_indices[0]
            peak_indice_2 = peak_indices[1]
            nadir_indice_1 = 0

            #NOTE: temporary solution to picking nadir; maybe better to get approx location of nadir through unsmoothed waveform
            if nadir_indices[0] < 100:
                nadir_indice_1 = nadir_indices[1]
            else:
                nadir_indice_1 = nadir_indices[0]

            #first rising edge
            time_rising_1 = time[0+nadir_clipping_length:peak_indice_1-peak_clipping_length+1]
            rising_1 = brf[0+nadir_clipping_length:peak_indice_1-peak_clipping_length+1]

            #first falling edge
            time_falling_1 = time[peak_indice_1+peak_clipping_length:nadir_indice_1-nadir_clipping_length+1]
            falling_1 = brf[peak_indice_1+peak_clipping_length:nadir_indice_1-nadir_clipping_length+1]

            #second rising edge
            time_rising_2 = time[nadir_indice_1+nadir_clipping_length:peak_indice_2-peak_clipping_length+1]
            rising_2 = brf[nadir_indice_1+nadir_clipping_length:peak_indice_2-peak_clipping_length+1]

            #second falling edge
            time_falling_2 = time[peak_indice_2+peak_clipping_length:len(brf)-nadir_clipping_length]
            falling_2 = brf[peak_indice_2+peak_clipping_length:len(brf)-nadir_clipping_length]

            #debugging
            ############################################################################################################
            linear_regressor.fit(time_rising_1.reshape(-1,1), rising_1.reshape(-1,1))
            y_rising_1 = linear_regressor.predict(time_rising_1.reshape(-1,1)).flatten()
            x_rising_1 = np.arange(0+nadir_clipping_length, peak_indice_1-peak_clipping_length+1, 1)

            linear_regressor.fit(time_falling_1.reshape(-1,1), falling_1.reshape(-1,1))
            y_falling_1 = linear_regressor.predict(time_falling_1.reshape(-1,1)).flatten()
            x_falling_1 = np.arange(peak_indice_1+peak_clipping_length, nadir_indice_1-nadir_clipping_length+1, 1)

            linear_regressor.fit(time_rising_2.reshape(-1,1), rising_2.reshape(-1,1))
            y_rising_2 = linear_regressor.predict(time_rising_2.reshape(-1,1)).flatten()
            x_rising_2 = np.arange(nadir_indice_1+nadir_clipping_length, peak_indice_2-peak_clipping_length+1, 1)

            linear_regressor.fit(time_falling_2.reshape(-1,1), falling_2.reshape(-1,1))
            y_falling_2 = linear_regressor.predict(time_falling_2.reshape(-1,1)).flatten()
            x_falling_2 = np.arange(peak_indice_2+peak_clipping_length, len(brf)-nadir_clipping_length, 1)
            ############################################################################################################

            cc_1 = np.corrcoef(rising_1, y_rising_1)
            cc_2 = np.corrcoef(falling_1, y_falling_1)
            cc_3 = np.corrcoef(rising_2, y_rising_2)
            cc_4 = np.corrcoef(falling_2, y_falling_2)

            if rising_or_falling == 'rising':
                set_x1(x_rising_1)
                set_y1(y_rising_1)
                
                set_x2(x_rising_2)
                set_y2(y_rising_2)

                return (cc_1[0][1] + cc_3[0][1])/2 #return the average of rising correlations

            elif rising_or_falling == 'falling':
                set_x1(x_falling_1)
                set_y1(y_falling_1)

                set_x2(x_falling_2)
                set_y2(y_falling_2)

                return (cc_2[0][1] + cc_4[0][1])/2 #return the average of falling correlations

    #single cycle isn't implemented
    #similar to linearity, although line is between peaks and nadirs
    #similar issue to linearity – has issues but also doesn't seem like a good overall feature
    def linearity_v2(time, brf, single_or_double, rising_or_falling):
        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]

        peak_indice_1 = peak_indices[0]
        peak_indice_2 = peak_indices[1]
        nadir_indice_1 = 0

        #NOTE: temporary solution to picking nadir; maybe better to get approx location of nadir through unsmoothed waveform
        if nadir_indices[0] < 100:
            nadir_indice_1 = nadir_indices[1]
        else:
            nadir_indice_1 = nadir_indices[0]

            #first rising edge
            x_rising_1 = np.arange(0, peak_indice_1+1, 1)
            y_rising_1 = np.linspace(brf[0], brf[peak_indice_1], peak_indice_1+1)
            rising_1 = brf[0:peak_indice_1+1]
            
            #first falling edge
            x_falling_1 = np.arange(peak_indice_1, nadir_indice_1+1, 1)
            y_falling_1 = np.linspace(brf[peak_indice_1], brf[nadir_indice_1], nadir_indice_1-peak_indice_1+1)
            falling_1 = brf[peak_indice_1:nadir_indice_1+1]

            #second rising edge
            x_rising_2 = np.arange(nadir_indice_1, peak_indice_2+1, 1)
            y_rising_2 = np.linspace(brf[nadir_indice_1], brf[peak_indice_2], peak_indice_2-nadir_indice_1+1)
            rising_2 = brf[nadir_indice_1:peak_indice_2+1]

            #second falling edge
            x_falling_2 = np.arange(peak_indice_2, len(brf), 1)
            y_falling_2 = np.linspace(brf[peak_indice_2], brf[len(brf)-1], len(brf)-peak_indice_2)
            falling_2 = brf[peak_indice_2:len(brf)]

            #normalize cross correlation values for each rising/falling interpolated line with the corresponding rising/falling edge
            cc_1 = np.corrcoef(rising_1, y_rising_1)
            cc_2 = np.corrcoef(falling_1, y_falling_1)
            cc_3 = np.corrcoef(rising_2, y_rising_2)
            cc_4 = np.corrcoef(falling_2, y_falling_2)

            if rising_or_falling == 'rising':
                set_x1(x_rising_1)
                set_y1(y_rising_1)
                
                set_x2(x_rising_2)
                set_y2(y_rising_2)

                avg_cc = (cc_1[0][1] + cc_3[0][1])/2

                return avg_cc #return the average of rising correlations

            elif rising_or_falling == 'falling':
                #debugging
                ######################################################
                set_x1(x_falling_1)
                set_y1(y_falling_1)

                set_x2(x_falling_2)
                set_y2(y_falling_2)

                set_pt1(peak_indice_1, brf[peak_indice_1])
                set_pt2(peak_indice_2, brf[peak_indice_2])
                ######################################################

                avg_cc = (cc_2[0][1] + cc_4[0][1])/2
                print(avg_cc)

                return avg_cc #return the average of falling correlations


    def slope(x, y, n):
        assert len(x) == len(y)
        numerator = n*np.sum(x*y) - np.sum(x)*np.sum(y)
        denominator = n*np.sum(x**2) - np.sum(x)**2
        #https://stackoverflow.com/questions/27784528/numpy-division-with-runtimewarning-invalid-value-encountered-in-double-scalars
        return numerator/denominator

    #angle of inflection on peaks and nadir; angle of inflection doesn't really make too much sense for single cycled waveforms?
    #the angle is between two cycles
    def angle_of_inflection(time, brf, single_or_double, peak_or_nadir):
        line_length = 100
        
        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]

        if single_or_double == 'single':
            peak_indice = peak_indices[0]
            pass

        elif single_or_double == 'double':
            peak_indice_1 = peak_indices[0]
            peak_indice_2 = peak_indices[1]
            nadir_indice_1 = 0

            if nadir_indices[0] < 100:
                nadir_indice_1 = nadir_indices[1]
            else:
                nadir_indice_1 = nadir_indices[0]

            new_time = np.linspace(0,2,len(time))

            #NOTE: using time from the oscilloscope data to calculate the angle
            #data for first peak
            time_pr1 = time[peak_indice_1-line_length+1:peak_indice_1+1]
            peak_rising_1 = brf[peak_indice_1-line_length+1:peak_indice_1+1]
            time_pf1 = time[peak_indice_1+1:peak_indice_1+line_length+1]
            peak_falling_1 = brf[peak_indice_1+1:peak_indice_1+line_length+1]

            #data for second peak
            time_pr2 = time[peak_indice_2-line_length+1:peak_indice_2+1]
            peak_rising_2 = brf[peak_indice_2-line_length+1:peak_indice_2+1]
            time_pf2 = time[peak_indice_2+1:peak_indice_2+line_length+1]
            peak_falling_2 = brf[peak_indice_2+1:peak_indice_2+line_length+1]

            #data for nadir
            time_nf1 = time[nadir_indice_1-line_length+1:nadir_indice_1+1]
            nadir_falling_1 = brf[nadir_indice_1-line_length+1:nadir_indice_1+1]
            time_nr1 = time[nadir_indice_1+1:nadir_indice_1+line_length+1]
            nadir_rising_1 = brf[nadir_indice_1+1:nadir_indice_1+line_length+1]

            #slopes for first peak
            peak_rising_slope_1 = brf_analysis.slope(time_pr1, peak_rising_1, line_length)
            peak_falling_slope_1 = brf_analysis.slope(time_pf1, peak_falling_1, line_length)

            #slopes for second peak
            peak_rising_slope_2 = brf_analysis.slope(time_pr2, peak_rising_2, line_length)
            peak_falling_slope_2 = brf_analysis.slope(time_pf2, peak_falling_2, line_length)
            
            #slopes for nadir
            nadir_falling_slope_1 = brf_analysis.slope(time_nf1, nadir_falling_1, line_length)
            nadir_rising_slope_1 = brf_analysis.slope(time_nr1, nadir_rising_1, line_length)

            #debugging/creating lines for visual debugging
            linear_regressor = LinearRegression()
            #approximated line for first falling edge
            linear_regressor.fit(time_nf1.reshape(-1,1), nadir_falling_1.reshape(-1,1))
            y_pred1 = linear_regressor.predict(time_nf1.reshape(-1,1))
            #approximated line for second rising edge
            linear_regressor.fit(time_nr1.reshape(-1,1), nadir_rising_1.reshape(-1,1))
            y_pred2 = linear_regressor.predict(time_nr1.reshape(-1,1))
            #getting corresponding x values to show in a plot; setting the corresponding x and y values of a line to global variables
            x1 = np.arange(nadir_indice_1-line_length+1, nadir_indice_1+1, 1)
            x2 = np.arange(nadir_indice_1+1, nadir_indice_1+line_length+1, 1)
            set_x1(x1)
            set_y1(y_pred1)
            set_x2(x2)
            set_y2(y_pred2)

            #calculating angles for peaks and nadir
            peak_angle_1 = math.atan(peak_rising_slope_1) - math.atan(peak_falling_slope_1)
            peak_angle_2 = math.atan(peak_rising_slope_2) - math.atan(peak_falling_slope_2)
            nadir_angle_1 = math.atan(nadir_falling_slope_1)+math.pi - math.atan(nadir_rising_slope_1)

            if peak_or_nadir == 'peak':
                return math.degrees((peak_angle_1+peak_angle_2)/2)
            elif peak_or_nadir == 'nadir':
                return math.degrees(nadir_angle_1)

    #ratio of approximate location of the peak within the cycle; just getting the indice and dividing by he cycle length
    def peak_location(brf, single_or_double):
        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]

        if single_or_double == 'single':
            peak_indice = peak_indices[0]
            ratio = peak_indice/len(brf)
            return ratio

        elif single_or_double == 'double':
            peak_indice_1 = peak_indices[0]
            peak_indice_2 = peak_indices[1]

            if nadir_indices[0] < 100:
                nadir_indice_1 = nadir_indices[1]
            else:
                nadir_indice_1 = nadir_indices[0]

            ratio_1 = peak_indice_1/nadir_indice_1
            ratio_2 = (peak_indice_2-nadir_indice_1)/(len(brf) - nadir_indice_1)

            set_pt1(peak_indice_1, brf[peak_indice_1])
            set_pt2(peak_indice_2, brf[peak_indice_2])
            
            return (ratio_1 + ratio_2)/2

    #cycle integral (so just the sum of the cycle), but need to divide by the cycle length to account of differences in cycle length
    #(integral/cycle length)
    def cycle_integral_avg(brf, single_or_double):
        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(brf, distance = 750)[0]
        
        if single_or_double == 'single':
            peak_indice = peak_indices[0]
            pass

        elif single_or_double == 'double':
            peak_indice_1 = peak_indices[0]
            peak_indice_2 = peak_indices[1]

            if nadir_indices[0] < 100:
                nadir_indice_1 = nadir_indices[1]
            else:
                nadir_indice_1 = nadir_indices[0]

            int_avg_1 = np.sum(brf[:nadir_indice_1])/nadir_indice_1
            int_avg_2 = np.sum(brf[nadir_indice_1:len(brf)])/(len(brf) - nadir_indice_1)

            return((int_avg_1 + int_avg_2)/2)

    #gradient analysis
    def brf_gradient_analysis(brf_database, single_or_double, save_path):
        cwd = os.getcwd()
        os.chdir(save_path)

        brf_database_list = database_processing.database_to_list(brf_database)

        comparison_list = []
        name_list = []
        type_list = []

        for i in range(len(brf_database_list)):
            folder_name = brf_database_list[i][0]
            brf_name = brf_database_list[i][1]
            bulb_type = brf_database_list[i][2]
            waveform_list = brf_extraction(folder_name, single_or_double).brf_list

            comparison_list.append(waveform_list[0])
            name_list.append(brf_name + ' (' + str(0) + ')')
            type_list.append(bulb_type)

            comparison_list.append(waveform_list[1])
            name_list.append(brf_name + ' (' + str(1) + ')')
            type_list.append(bulb_type)

        assert len(comparison_list) == len(name_list) and len(name_list) == len(type_list)

        for i in range(len(comparison_list)):
            print(name_list[i])
            smoothed_brf = raw_waveform_processing.savgol(comparison_list[i], savgol_window)
            averaged_brf = raw_waveform_processing.moving_average(smoothed_brf, mov_avg_w_size)
            brf_analysis.gradient(averaged_brf, name_list[i])
            print(f'{name_list[i]} + done')

        os.chdir(cwd)

            

class brf_classification():
    #Sheinin et al. BRF comparison method 
    def compare_brfs(brf_database, num_comparisons, single_or_double): #num_comparisons is the number of comparisons we want to make (e.g. 3)
        bulb_types = brf_database.Bulb_Type.unique()
        brf_database_list = database_processing.database_to_list(brf_database)
        
        ground_list = list([])
        predicted_list = list([])

        bulb_type_list = brf_database.Bulb_Type.unique()

        #Personal note: if want to generate confusion matrix for bulb type, don't include outer bulb_type for loop
        #               otherwise, if want to compare unique BRFs, compare between the bulb type classification
        #               additionally, need to change what is being appended to the error list (type or name?)

        for bulb_type in bulb_type_list:
            print(bulb_type)
            same_type_database = database_processing.same_type_df(brf_database,str(bulb_type))
            same_type_database_list = database_processing.database_to_list(same_type_database)
            same_type_name_list = database_processing.database_to_list(same_type_database.Name)
            #RENAMING same_type_database_list to brf_database_list for convenience
            brf_database_list = same_type_database_list

            #unindent this if comparing bulb types for the ENTIRE database 
            for i in range(len(brf_database_list)): #outer loop for comparing against specific bulb
                brf_1 = brf_extraction(brf_database_list[i][0], single_or_double).brf_list[0] #grabs first BRF in list
                brf_name_1 = brf_database_list[i][1]
                brf_type_1 = brf_database_list[i][2]
                error_list = list([])
                print(brf_name_1)
                for j in range(len(brf_database_list)): #inner loop that goes through entire list
                    for k in range(1, num_comparisons+1):
                        brf_2 = brf_extraction(brf_database_list[j][0], single_or_double).brf_list[k]
                        brf_name_2 = brf_database_list[j][1]
                        brf_type_2 = brf_database_list[j][2]
                        error_score = brf_analysis.min_error(brf_1, brf_2)
                        error_list.append((error_score, brf_name_1, brf_name_2))

                    '''
                    0 --> error score
                    1 --> brf_1
                    2 --> brf_2
                    sorted by min error, so grab n closest matches
                    '''

                #sorting by error score
                sorted_min_error = sorted(error_list, key = lambda x: x[0])

                #grabs top three closest matches
                for k in range(1, num_comparisons+1):
                    ground_list.append(sorted_min_error[k][1])
                    predicted_list.append(sorted_min_error[k][2])

            #PLOT FOR UNIQUE BRFS WITHIN A BULB TYPE CATEGORY
            # plots.confusion_matrix_type(ground_list, predicted_list, bulb_types)
            plots.confusion_matrix_unique(ground_list, predicted_list, same_type_name_list, f'Confusion Matrix of For {bulb_type} Bulb Type Using Sheinin\'s Statistical Measurement \n(tested with {num_comparisons} double cycles)')
            #need to reset lists
            ground_list = list([])
            predicted_list = list([])
            print()

    '''
    processes the input and output list for training, in addition to the prediction_list (aka testing set)

    INPUT:
        pkl_path                path for pickle file
        type                    bulb type; used if the confusion matrix is for unique BRFs w.r.t. a bulb type
        classification_type     'name' or 'type' classification
        num_test_waveforms      number of waveforms to form the test set (it not using k-fold cross validation)
        num_features            number of features used for KNN; used for assertion statement
        k_fold_CV               True or False for whether to use k-fold or not
        split_number            for setting the k-fold cross validation seed

    OUTPUT:
        KNN_input               training input for KNN
        KNN_output              training output for KNN
        KNN_prediction_list     test set for KNN

    THIS METHOD NEEDS TO BE CHANGED TO JUST INCLUDE THE BULB TYPE AND NAME OUTPUT IN KNN_prediction_list
    '''

    def KNN_in_out_pkl(pkl_path, type, classification_type, num_test_waveforms, num_features, k_fold_CV, split_number):
        if k_fold_CV:
            #for now, only using k-fold with 'type'; identifying BRFs uniquely isn't ready yet
            assert classification_type == 'type'

            KNN_df = pd.read_pickle(pkl_path)
            name_list = KNN_df.Name.unique()
            type_list = list()
            feature_list = list()
            for name in name_list:
                name_df = KNN_df.loc[KNN_df['Name'] == name]
                bulb_type = name_df.Bulb_Type.tolist()[0] #grab the first item; they all should be the same
                feature_list.append(list(name_df.Features.tolist()))
                type_list.append(bulb_type)

            assert len(name_list) == len(feature_list)

            unique_type_list = list(set(type_list))
            
            '''
            temp_input_train        array of arrays; [[[num_features], [num_features], ... , [num_features]], ...
                                                        , [[num_features], [num_features], ... , [num_features]]]
                                    [num_features] is an array of length num_features
            temp_output_train       zipped list of list of name_list and type_list

            temp_input_test         format is same as temp_input_train, but size is smaller
            temp_output_test        zipped list of list of name_list and type_list
            '''

            #order for classification_list is (name, type)
            classification_list = list(zip(name_list, type_list))

            temp_input_train, temp_input_test, temp_output_train, temp_output_test = train_test_split(feature_list, classification_list, test_size=0.2, random_state=split_number)
            # temp_input_train, temp_input_test, temp_output_train, temp_output_test = train_test_split(feature_list, type_list, test_size=0.2, random_state=2)
            
            output_train_unzipped = list(zip(*temp_output_train))
            output_test_unzipped = list(zip(*temp_output_test))

            # for now, only using bulb types (hence 'output_train_unzipped[1]')
            temp_output_train = output_train_unzipped[1]
            temp_output_test = output_test_unzipped[1]
            
            temp_output_train_name = output_train_unzipped[0]
            temp_output_test_name = output_test_unzipped[0]

            assert list(set(temp_output_train_name) & set(temp_output_test_name)) == []
            assert len(temp_input_train) == len(temp_output_train)
            assert len(temp_input_test) == len(temp_output_test)

            KNN_input_train = list()
            KNN_output_train = list()

            KNN_input_test = list()
            KNN_output_test = list()
            KNN_prediction_name_list = list()

            for i in range(len(temp_input_train)):
                num_items = len(temp_input_train[i])
                KNN_input_train.extend(temp_input_train[i])
                KNN_output_train.extend(list([temp_output_train[i]]) * num_items)

            for i in range(len(temp_input_test)):
                num_items = len(temp_input_test[i])
                KNN_input_test.extend(temp_input_test[i])
                KNN_output_test.extend(list([temp_output_test[i]]) * num_items)
                KNN_prediction_name_list.extend(list([temp_output_test_name[i]]) * num_items)

            assert len(KNN_input_train) == len(KNN_output_train)
            assert len(KNN_input_test) == len(KNN_output_test)

            # print(f'Number of training samples: {len(KNN_output_train)}')

            # KNN_input, KNN_output = database_processing.load_KNN_pkl(pkl_path, classification_type)
            # KNN_input_train, KNN_input_test, KNN_output_train, KNN_output_test = train_test_split(KNN_input, KNN_output, test_size=0.2, random_state=0)
            
            KNN_prediction_list = list(zip(KNN_input_test, KNN_output_test, KNN_prediction_name_list))
            return KNN_input_train, KNN_output_train, KNN_prediction_list
        else:
            KNN_df = pd.read_pickle(pkl_path)

            if type != None:
                KNN_df = KNN_df.loc[KNN_df['Bulb_Type'] == type]

            KNN_input = list()
            KNN_output = list()

            input_param_list = list()
            output_param_list = list()
            name_param_list = list()

            unique_outputs = KNN_df.Name.unique()

            for output in unique_outputs:
                brf_df = KNN_df.loc[KNN_df['Name'] == output]

                input_list = brf_df.Features.tolist()
                type_list = brf_df.Bulb_Type.tolist()
                name_list = brf_df.Name.tolist()

                assert len(input_list) == len(type_list)
                assert len(type_list) == len(name_list)

                for i in range(len(input_list)):
                    if i < num_test_waveforms: #determines number of test/training waveforms
                        if classification_type == 'name':
                            input_param_list.append(input_list[i])
                            output_param_list.append(name_list[i])
                            name_param_list.append(name_list[i])
                            # KNN_prediction_list.append([input_param, brf_name])
                        elif classification_type == 'type':
                            input_param_list.append(input_list[i])
                            output_param_list.append(type_list[i])
                            name_param_list.append(name_list[i])
                            # KNN_prediction_list.append([input_param, bulb_type])
                    else:
                        KNN_input.append(input_list[i])
                        if classification_type == 'name':
                            KNN_output.append(name_list[i])
                        if classification_type == 'type':
                            KNN_output.append(type_list[i])

            KNN_input = np.vstack((KNN_input))
            KNN_prediction_list = list(zip(input_param_list, output_param_list, name_param_list))

            assert len(KNN_input[0]) == num_features

            return KNN_input, KNN_output, KNN_prediction_list

    #classification_type is for either the BRF name or BRF type; options are either 'name' or 'type'
    def train_KNN(KNN_input, KNN_output, number_neighbors, num_features, weights):

        # https://stackoverflow.com/questions/50064632/weighted-distance-in-sklearn-knn
        #angle of inflection, integral ratio, integral average, peak location, crest factor, kurtosis, skew
        # weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # weights = np.array([0, 0, 0, 0, 1, 1, 1])
        # weights = np.array([0.001, 0.001, 0.001, 0.001, 100, 100, 100])
        # weights = np.array([0, 0, 0, 1, 1, 1, 1])

        #try more integer values
        #small grid search (e.g. 5, 10, 15)
        #try to set weights so that the "best" feature weighs more and worse is less and vice versa to prove that the weights are doing something

        assert len(weights) == num_features

        brf_KNN_model = KNeighborsClassifier(n_neighbors = number_neighbors, p = 2, metric = 'wminkowski', metric_params = {'w': weights}) #used 9 because about 9 waveforms for each BRF
        # brf_KNN_model = KNeighborsClassifier(n_neighbors = number_neighbors, metric = 'minkowski') #used 9 because about 9 waveforms for each BRF
        # brf_KNN_model = KNeighborsClassifier(n_neighbors = number_neighbors) #used 9 because about 9 waveforms for each BRF
        # brf_KNN_model = KNeighborsClassifier(n_neighbors = number_neighbors, weights = 'distance') #used 9 because about 9 waveforms for each BRF

        brf_KNN_model.fit(KNN_input, KNN_output)

        return brf_KNN_model

    #name_k is for bulb identification, which I am not doing at the moment!
    def KNN(brf_database, KNN_in, KNN_out, KNN_prediction_list, number_neighbors, classification_type, num_test_waveforms, num_features, weights, Entire, name_k, MisClass):
        #this assert statement should always hold because the pairing does not make sense
        if classification_type == 'type':
            assert Entire == True

        no_match = True

        # grabs first bulb
        bulb_type = brf_database.Bulb_Type.tolist()[0]
        # same_type_database = database_processing.same_type_df(brf_database,str(bulb_type))
        name_list = database_processing.database_to_list(brf_database.Name)

        #requires and input list and output list to train the model
        # classification_list is either:
        # 1) all unique BRFs, or
        # 2) Bulb types
        if Entire:
            brf_KNN_model = brf_classification.train_KNN(KNN_in, KNN_out, number_neighbors, num_features, weights)
            if classification_type == 'name':
                classification_list = database_processing.database_to_list(brf_database.Name)
            elif classification_type == 'type':
                classification_list = brf_database.Bulb_Type.unique()
        else:
            brf_KNN_model = brf_classification.train_KNN(KNN_in, KNN_out, number_neighbors, num_features, weights)

        if MisClass and classification_type == 'type':
            misclassification_array = np.zeros(len(name_list))

        ground_list = list([])
        predicted_list = list([])

        true_positive = 0
        total = len(KNN_prediction_list)

        #this is for misclassification analysis; analysis only done on *bulb type* labels
        if classification_type == 'type':
            type_list = list(brf_database.Bulb_Type.unique())
            true_positive_list = np.arange(0,len(type_list),1)
            total_list = np.arange(0,len(type_list),1)

        '''
        index 0: feature array
        index 1: classification type ('name' or bulb 'type'); this is more important if classification_type is 'type'
        index 2: brf name
        '''
        for prediction in KNN_prediction_list:
            input_data = prediction[0]
            expected_output = prediction[1]
            brf_name = prediction[2] #this might be wrong...? have this here just for the future (in case I need it)

            output = brf_KNN_model.predict([input_data])[0]

            #don't evaluate k closest neighbors with bulb type classification
            if classification_type == 'type':
                if expected_output == output:
                    true_positive += 1
                    true_positive_list_index = type_list.index(output)
                    true_positive_list[true_positive_list_index] = true_positive_list[true_positive_list_index] + 1
                total_list_index = type_list.index(expected_output)
                total_list[total_list_index] = total_list[total_list_index] + 1

                ground_list.append(expected_output)
                predicted_list.append(output)
                continue

            neighbor_indices = brf_KNN_model.kneighbors([input_data])[1][0]

            neighbor_indices = neighbor_indices[:name_k]

            # print(f'{expected_output}; Index = {index}')
            for model_index in neighbor_indices:
                if Entire:
                    classification_list_index = brf_KNN_model._y[model_index]
                    if classification_list[classification_list_index] == expected_output:
                        true_positive += 1
                        ground_list.append(expected_output)
                        predicted_list.append(expected_output)
                        no_match = False
                        break
                else:
                    name_list_index = brf_KNN_model._y[model_index]
                    if name_list[name_list_index] == expected_output:
                        true_positive += 1
                        ground_list.append(expected_output)
                        predicted_list.append(expected_output)
                        no_match = False
                        break
            if no_match:
                #if not amongst the closest neighbors, then check if the "predict" method predicts the correct result
                if expected_output == output:
                    true_positive += 1
                else:
                    if MisClass and classification_type == 'type':
                        index_output = name_list.index(brf_name)
                        misclassification_array[index_output] = misclassification_array[index_output] + 1
                ground_list.append(expected_output)
                predicted_list.append(output)

            no_match = True

        # this should be accuracy...?
        accuracy = true_positive/total
        print(f'Accuracy: {accuracy}')


        if not skip:
            print('Do you want to skip plotting? [y]/other')
            val = input()
            if val == 'y':
                skip_plots(True)

        if not skip:
            if Entire:
                if classification_type == 'name':
                    plots.confusion_matrix_unique(ground_list, predicted_list, classification_list, f'[int_avg, peak_loc, crest_factor, kurtosis, skew]\nConfusion Matrix for Entire Database Using KNN\nWeights: {weights}\nClosest {num_test_waveforms} Neighbors & KNN Prediction\n(~7 double cycles for training, 3 for testing)\nOverall Correctness: {accuracy}')
                elif classification_type == 'type':
                    plots.confusion_matrix_type(ground_list, predicted_list, classification_list, f'[int_avg, peak_loc, crest_factor, kurtosis, skew]\nConfusion Matrix of Bulb Types Using KNN and k-fold CV (80% Train/20% Test)\nWeights: {weights}\nClosest {num_test_waveforms} Neighbors & KNN Prediction\n(~7 double cycles for training, 3 for testing)\nOverall Correctness: {accuracy}')
            else:
                plots.confusion_matrix_unique(ground_list, predicted_list, name_list, f'[angle, integral_ratio, int_avg, peak_loc, crest_factor, kurtosis, skew]\nWeights: {weights}\nConfusion Matrix for {bulb_type} Bulb Type Using KNN\nClosest {num_test_waveforms} Neighbors & KNN Prediction\n(~7 double cycles for training, 3 for testing)\nOverall Correctness: {accuracy}')

        if MisClass and classification_type == 'type':
            return misclassification_array, true_positive, total, true_positive_list, total_list #lists refer to bulb types        

    #get different KNN_in, KNN_out, KNN_prediction_list, and then RUN THE METHOD
    def KNN_analysis_pkl(pkl_path, brf_database, classification_type, weights, Entire, num_test_waveforms, num_features, number_neighbors, name_k):
        # other parameters
        ########################################################################################################################################################
        MisClass = False            #no misclassification analysis done with this method, so set MisClass to False
        ########################################################################################################################################################

        if Entire:
            KNN_in, KNN_out, KNN_prediction_list = brf_classification.KNN_in_out_pkl(pkl_path, None, classification_type, num_test_waveforms, num_features, k_fold_CV=False, split_number=0)
            brf_classification.KNN(brf_database, KNN_in, KNN_out, KNN_prediction_list, number_neighbors, classification_type, num_test_waveforms, num_features, weights, Entire, name_k, MisClass)
        else:
            bulb_type_list = database_processing.return_bulb_types(brf_database)
            for bulb_type in bulb_type_list:
                print(bulb_type)
                same_type_database = database_processing.same_type_df(brf_database,str(bulb_type))
                KNN_in, KNN_out, KNN_prediction_list = brf_classification.KNN_in_out_pkl(pkl_path, bulb_type, classification_type, num_test_waveforms, num_features, k_fold_CV=False, split_number=0)
                brf_classification.KNN(same_type_database, KNN_in, KNN_out, KNN_prediction_list, number_neighbors, classification_type, num_test_waveforms, num_features, weights, Entire, MisClass)
                print()

    '''
    DESCRIPTION:
        analysis of bulb types ONLY; evaluates ___

    INPUT:

    OUTPUT:
    '''
    def k_fold_analysis(pkl_path, brf_database, classification_type, weights, num_test_waveforms, num_features, min_num_neighbors, max_num_neighbors, num_splits, MisClass):
        #other parameters
        ########################################################################################################################################################
        k_fold_CV = True            #set k_fold_CV to true because evaluating misclassification from using k-fold cross validation
        name_k = 0                  #set name_k to 0 because only evaluating bulb types, not nearest name_k neighbors for the brf-name-based KNN model
        Entire = True               #set Entire to True because when evaluating bulb type, all the bulbs (the entire bulb set) will be evaluated
        ########################################################################################################################################################

        name_list = database_processing.database_to_list(brf_database.Name)
        type_list = brf_database.Bulb_Type.unique()
        k_fold_misclassification = np.zeros(len(name_list))

        type_true_positive = np.zeros(len(type_list))
        type_total = np.zeros(len(type_list))

        num_true_positive = 0
        total = 0
        random_state = 0
        split_number = 0

        overall_accuracy = list()
        cfl_accuracy = list()
        halogen_accuracy = list()
        incandescent_accuracy = list()
        led_accuracy = list()

        #TODO: need to use 'MisClass' variable in this method

        k_list = np.arange(min_num_neighbors,max_num_neighbors+1,1)
        for number_neighbors in k_list:
            print(f'Number of Neighbors: {number_neighbors}')
            while split_number < num_splits:
                # print(f'Split iteration: {split_number}')
                # print(f'Split Number: {split_number + 1}')

                KNN_in, KNN_out, KNN_prediction_list = brf_classification.KNN_in_out_pkl(pkl_path, None, classification_type, num_test_waveforms, num_features, k_fold_CV, random_state)
                if classification_type == 'type':
                    '''
                    index 0: feature array
                    index 1: classification type ('name' or bulb 'type'); this is more important if classification_type is 'type'
                    index 2: brf name
                    '''
                    #gets the list of (unique) bulb types --> should be CFL, incandescent, halogen, LED (not sure if in that order)
                    unique_type_list = list(set(list(zip(*KNN_prediction_list))[1]))
                    if len(unique_type_list) < 4:
                        random_state = random_state + 1
                        continue
                
                if classification_type == 'type':
                    misclassification_array, k_true_positive, k_total, k_true_positive_list, k_total_list = brf_classification.KNN(brf_database, KNN_in, KNN_out, KNN_prediction_list, number_neighbors, classification_type, num_test_waveforms, num_features, weights, Entire, name_k, MisClass)

                    # print('Accumulative misclassification')
                    k_fold_misclassification = k_fold_misclassification + misclassification_array
                    # plots.misclass_bar_graph(name_list, k_fold_misclassification, 'Misclassification Bar Graph')

                    type_true_positive = type_true_positive + k_true_positive_list
                    type_total = type_total + k_total_list

                    num_true_positive = num_true_positive + k_true_positive
                    total = total + k_total
                    random_state = random_state + 1
                    split_number = split_number + 1

            if MisClass:
                # plots.misclass_bar_graph(name_list, k_fold_misclassification, 'Misclassification Bar Graph')
                total_accuracy = num_true_positive / total
                print(f'Accuracy for k = {number_neighbors} and {num_splits} folds: {total_accuracy}')

                type_accuracy = type_true_positive / type_total
                for i in range(len(type_accuracy)):
                    print(f'{type_list[i]}: {type_accuracy[i]}')
                print()

                overall_accuracy.append(total_accuracy)
                cfl_accuracy.append(type_accuracy[0])
                halogen_accuracy.append(type_accuracy[1])
                incandescent_accuracy.append(type_accuracy[2])
                led_accuracy.append(type_accuracy[3])

                k_fold_misclassification = np.zeros(len(name_list))
                num_true_positive = 0
                total = 0
                random_state = 0
                split_number = 0

        title = f'Bulb Type KNN Model: Accuracy Using k-fold Cross Validation with {num_splits} Splits'
        plots.k_and_k_fold(k_list, overall_accuracy, cfl_accuracy, halogen_accuracy, incandescent_accuracy, led_accuracy, title)

    '''
    THIS METHOD IS OUTDATED
    uses old "KNN_in_out" method
    '''
    def grid_search(brf_database, number_neighbors, classification_type, num_test_waveforms, single_or_double, num_features, end_weight, step_length, Tallied, Entire):
        num_best = 10
        weights = np.arange(0, end_weight+step_length, step_length)
        weights_list = list()
        
        KNN_in, KNN_out, KNN_prediction_list = brf_classification.KNN_in_out(brf_database, number_neighbors, classification_type, num_test_waveforms, single_or_double, num_features)
        print()
        
        for i in range(num_features):
            weights_list.append(weights)
        weight_combinations = list(product(*weights_list))
        accuracy_list = list(np.zeros((num_best)))
        weights_list = list(np.zeros((num_best)))
        for i in range(len(weight_combinations)):
            weights = weight_combinations[i]
            accuracy = brf_classification.KNN(brf_database, KNN_in, KNN_out, KNN_prediction_list, number_neighbors, classification_type, num_test_waveforms, single_or_double, num_features, weights, Tallied, Entire, True)
            for j in range(len(accuracy_list)):
                if accuracy > accuracy_list[j]:
                    accuracy_list[j] = accuracy
                    weights_list[j] = weights
                    break
            print(f'Accuracy: {accuracy}')
            print(f'{i+1}/{len(weight_combinations)} Combinations Completed')
            print()
        for i in range(len(accuracy_list)):
            print(f'Accuracy: {accuracy_list[i]}')
            print(f'Weights: {weights_list[i]}')
            print()

if __name__ == "__main__":
    csv_pkl_save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database'
    # pkl_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\KNN.pkl'
    pkl_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\KNN_downsized.pkl'
    # pkl_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\KNN_downsized_smoothed.pkl'
    # pkl_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\Zeal_stats.pkl'
    brf_database = database_processing(database_path).brf_database

    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'halco_led_10w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_6p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_8w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_7p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_12w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_5p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_11w'].index)

    #taking out these BRFs increased accuracy
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'halco_halogenxenon_39w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_halogenincandescent_65w'].index)

    # brf_classification.compare_brfs(brf_database, 3, 'double')
    
    #'Entire' is for the entire database
    # weights = np.array([0.25, 0.0, 0.75, 0.5, 0.75, 1.0, 1.0])
    # weights = np.array([0.0, 0.75, 0.5, 0.75, 1.0, 1.0])
    # weights = np.array([0.75, 0.5, 0.75, 1.0, 1.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # pickles features
    # database_processing.pkl_KNN_in_out(brf_database, 'double', 5, csv_pkl_save_path)

    # runs KNN analysis on pkl file
    brf_classification.KNN_analysis_pkl(pkl_path, brf_database, 'type', weights, Entire = True, num_test_waveforms=3, number_neighbors=6, num_features=5, name_k=3)
    #after choosing K, train on train+validation set; final with test set

    # k-fold analysis
    # brf_classification.k_fold_analysis(pkl_path, brf_database, 'type', weights, num_test_waveforms = 999, num_features = 5, min_num_neighbors = 2, max_num_neighbors = 10, num_splits = 12, MisClass = True)
    
    # brf_classification.grid_search(brf_database, 3, 'name', 3, 'double', 7, end_weight = 1, step_length = 0.25, Tallied = False, Entire = True)

    # df = pd.read_pickle(pkl_path)
    # df_list = df.values.tolist()
    # print(len(df_list[0]))

    '''
    # brf_analysis.brf_gradient_analysis(brf_database, 'double', gradient_save_path)
    # brf_analysis.test_analysis_method(brf_database, 'integral_ratio', 'double', 'TCP LED 9.5W')
    # brf_analysis.test_analysis_method(brf_database, 'linearity_v2', 'double')
    # brf_analysis.test_analysis_method(brf_database, 'test', 'single')
    # brf_classification.PCA_KNN(brf_database, 'double', 7, 10, 3)
    # #export stats to csv
    # method_name_list = ['Integral_Ratio', 'Nadir_Angle', 'Crest_Factor', 'Kurtosis', 'Skew']
    # database_processing.export_all_to_csv(brf_database, method_name_list, 'double')
    # #generate stats figures with 95% CIs
    # brf_analysis.feature_analysis(brf_database, method_name_list, 'double')
    '''