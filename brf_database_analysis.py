import time
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
import os
import seaborn as sn


#CODE IS GETTING MESSY; NEED TO DEDICATE TIME TO CLEAN UP

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

def set_falling_slope_var(val):
    global falling_slope
    falling_slope = val

def set_rising_slope_var(val):
    global rising_slope
    rising_slope = val

def set_nadir_var(val):
    global nadir
    nadir = val

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

def set_pt1(x, y):
    global x_pt1
    global y_pt1

    x_pt1 = x
    y_pt1 = y

############################################################

def clear_lists(list0, list1, list2, list3, list4, list5, list6, list7, list8, list9, list10, list11):
    list0.clear()
    list1.clear()
    list2.clear()
    list3.clear()
    list4.clear()
    list5.clear()
    list6.clear()
    list7.clear()
    list8.clear()
    list9.clear()
    list10.clear()
    list11.clear()

class plots():
    def show_plot(waveform):
        plt.plot(waveform)
        plt.show()
    
    def save_plot(waveform, plot_name):
        plt.plot(waveform)
        plt.title(plot_name)
        plt.savefig(plot_name + '.png')
        plt.clf()

    def save_gradient_plot(orig_brf, gradient, plot_name):
        fig, ax = plt.subplots(1, 2, constrained_layout = True)
        ax[0].plot(orig_brf)
        ax[0].set_title('Original, Smoothed BRF')

        ax[1].plot(gradient)
        ax[1].set_title('Smoothed Gradient')

        fig.suptitle(plot_name)
        fig.set_size_inches(12,6)
        plt.savefig(plot_name + '.png')
        plt.clf()

    def confusion_matrix_type(ground_list, predicted_list, bulb_types):
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
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)
        plt.title('Confusion Matrix of Bulb Types')
        plt.xlabel('Predicted', fontsize = 16)
        plt.ylabel('Expected', fontsize = 16)
        plt.show()

    #made another (redundant) method because I remove two "bulb types" (Halogen-Incandescent and Halogen-Xenon)
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
        plt.xticks(rotation = 45, ha = 'right') #why is alignment 'right'...?
        plt.ylabel('Expected', fontsize = 16)
        plt.yticks(rotation = 45, va = 'top') #why is alignment 'top'...?
        plt.tight_layout()
        plt.show()

    def KNN_confusion_matrix_tallies(tallied_matrix, total_matrix, unique_brf_names, title):
        
        assert tallied_matrix.shape == total_matrix.shape

        confusion_matrix = np.divide(tallied_matrix, total_matrix)

        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in unique_brf_names], columns = [i for i in unique_brf_names])
        plt.figure(figsize = (8,7)).tight_layout()
        sn.heatmap(df_cm)
        plt.title(title)
        plt.xlabel('Predicted', fontsize = 16)
        plt.xticks(rotation = 45, ha = 'right')
        plt.ylabel('Expected', fontsize = 16)
        plt.yticks(rotation = 45, va = 'top')
        plt.tight_layout()
        plt.show()

    def feature_analysis_bar_graphs(x_labels, mean_list, CI_list, plot_title, save_path):
        plt.figure(figsize = (15,8))
        plt.bar(x_labels, mean_list)
        plt.errorbar(x_labels, mean_list, yerr = CI_list, fmt = 'o', color = 'r')
        plt.xticks(rotation = 45, ha = 'right')
        plt.tight_layout()
        plt.title(plot_title)
        plt.savefig(save_path)
        plt.close()


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

    def return_names(brf_database):
        return brf_database.Name

    def return_bulb_types(brf_database):
        return brf_database.Bulb_Type.unique()

    def display_types(brf_database):
        print(brf_database.Bulb_Type.unique())

    def return_bulb_type_waveforms(brf_database, bulb_type):
        return brf_database.loc[brf_database['Bulb_Type'] == bulb_type]

    def return_all_waveforms(brf_database):
        return brf_database['Folder_Name']

    def drop_bulb_type_column(brf_database):
        return brf_database.drop('Bulb_Type', axis = 1)

    #ordered by: [folder name, name, bulb_type]
    def database_to_list(brf_database):
        return brf_database.values.tolist()

    def lists_to_csv(file_name, name_list_1, list_1, name_list_2, list_2):
        d = {name_list_1: list_1, name_list_2: list_2}
        df = pd.DataFrame(data = d)
        save_path = brf_analysis_save_path + '\\' + file_name
        df.to_csv(save_path)

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

    def return_mean_CI_lists(df, method_name_list):
        mean_list = list([])
        CI95_list = list([])

        # print(df)

        for method_name in method_name_list:
            values = df[method_name].tolist()
            mean = np.mean(values)
            std = np.std(values)
            CI95 = 1.96*std/math.sqrt(len(values))

            mean_list.append(mean)
            CI95_list.append(CI95)

        return mean_list, CI95_list

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

#NOTE: 750 IS HARDCODED; NEEDS TO BE CHANGED/AUTOMATED EVENTUALLY
    def clean_brf(time, brf):
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]
        corresponding_time = time[nadir_indices[0]:nadir_indices[2]]
        cleaned_brf = brf[nadir_indices[0]:nadir_indices[2]] #first and third nadir
        return corresponding_time, cleaned_brf

    def normalize(array):
        min = np.amin(array)
        max = np.amax(array)
        normalized = (array-min)/(max-min)
        return normalized

    def savgol(brf, savgol_window):
        smoothed = signal.savgol_filter(brf, savgol_window, 3)
        return smoothed

    def moving_average(data, window_size):
        return signal.convolve(data, np.ones(window_size) , mode = 'valid') / window_size

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
    def test_analysis_method(brf_database, method_name, single_or_double):
        brf_database_list = database_processing.database_to_list(brf_database)

        for i in range(len(brf_database_list)):
            smoothed = None
            folder_name = brf_database_list[i][0]
            brf_name = brf_database_list[i][1]
            bulb_type = brf_database_list[i][2]
            extracted_lists = brf_extraction(folder_name, single_or_double)
            time_list = extracted_lists.time_list
            waveform_list = extracted_lists.brf_list
            print(brf_name)
            mean = 0
            values = np.array([])
            std = 0

            #below is for writing specific tests for a method
            if method_name == 'integral_ratio':
                for j in range(len(waveform_list)): #first 3 waveforms
                    # smoothed = raw_waveform_processing.savgol(waveform_list[j], savgol_window)
                    smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[j], savgol_window), mov_avg_w_size)
                    ratio = brf_analysis.integral_ratio(smoothed, single_or_double)
                    values = np.hstack((values, ratio))
                    # print(ratio)
                    # print()
                    # plots.show_plot(smoothed)

            elif method_name == 'linearity':
                for j in range(len(waveform_list)):
                    smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[j], savgol_window), mov_avg_w_size)
                    correlation = brf_analysis.linearity(time_list[j], smoothed, single_or_double, 'falling')
                    values = np.hstack((values, correlation))
                    # print(correlation)
                    # print()
                    # plots.show_plot(smoothed)

            elif method_name == 'angle_of_inflection':
                for j in range(len(waveform_list)):
                    smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[j], savgol_window), mov_avg_w_size)
                    angle = brf_analysis.angle_of_inflection(time_list[j], smoothed, single_or_double, 'nadir')
                    values = np.hstack((values, angle))
                    # print(angle)
                    # print()
                    # plots.show_plot(smoothed)
            # print(falling_slope)
            # print(rising_slope)
            # print(nadir)

            elif method_name == 'test':
                length_list = np.array([])
                for j in range(len(waveform_list)):
                    length_list = np.hstack((length_list, len(waveform_list[j])))

                min_length = int(np.amin(length_list))

                brf_list = np.ones((min_length))
                for j in range(len(waveform_list)):
                    brf = np.array([waveform_list[j][:min_length]])
                    brf_list = np.vstack((brf_list, brf))
                
                brf_list = brf_list[1:len(brf_list)]
                brf_analysis.PCA(brf_list, 25)
                quit()

            mean = np.sum(values)/len(values)
            std = math.sqrt(np.sum((values-mean)**2/(len(values)-1)))

            print(mean)
            print(std) #smaller the better/more reliable
            print()

            plt.plot(smoothed)
            # plt.plot(x1, y1, color = 'red')
            # plt.plot(x2, y2, color = 'red')
            plt.show()

    def min_error(brf_1, brf_2):
        brf_1, brf_2 = raw_waveform_processing.truncate_longer(brf_1, brf_2)
        error = np.sum(np.square(np.absolute(np.array(brf_1) - np.array(brf_2))))
        return error

    def crest_factor(brf):
        peak_value = np.amax(brf)
        rms = math.sqrt(np.sum(np.array(brf))/len(brf))
        crest_factor = peak_value/rms
        return crest_factor

    def kurtosis(brf):
        return stats.kurtosis(brf)

    def skew(brf):
        return stats.skew(brf)

    def gradient(data, name):
        horizontal_gradient = np.array([1, 0, -1])
        # horizontal_gradient = np.array([1/12, -2/3, 0, 2/3, -1/12])
        gradient_x = signal.convolve(data, horizontal_gradient, mode = 'valid')
        smoothed_gradient = raw_waveform_processing.savgol(gradient_x, savgol_window)
        # plots.save_gradient_plot(data, smoothed_gradient, name)
        plots.save_gradient_plot(data, gradient_x, name)
        return gradient_x

    #this might not be the one we want (if we use NCC)
    def NCC(data_1, data_2):
        mean_subtracted_1 = data_1 - np.mean(data_1)
        min_1 = np.amin(mean_subtracted_1)
        max_1 = np.amax(mean_subtracted_1)
        norm_1 = mean_subtracted_1/(np.sqrt(np.sum((data_1 - np.mean_data_1)**2)))

        mean_subtracted_2 = data_2 - np.mean(data_2)
        min_2 = np.amin(mean_subtracted_2)
        max_2 = np.amax(mean_subtracted_2)
        norm_2 = mean_subtracted_1/(np.sqrt(np.sum((data_2 - np.mean_data_2)**2)))

        return np.dot(norm_1, norm_2)

    #this ratio is defined by the left side of the peak/right side of the peak (rising/falling)
    '''
    if ratio == 1, then peak is roughly in the middle
    if ratio < 1, then peak is skewed left
    if ratio > 1, then peak is skewed right
    '''

    def normalize_cycle(brf): #integral, or sum, is equal to 1
        integral = np.sum(brf)
        normalized = brf/integral
        return normalized

    def integral_ratio(brf, single_or_double):
        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]

        # print(f'Peaks: {peak_indices}')
        # print(f'Nadirs: {nadir_indices}')

        if single_or_double == 'single':
            peak_indice = peak_indices[0]

            rising = brf[0:peak_indice+1]
            falling = brf[peak_indice:len(brf)]

            ratio = np.sum(rising)/np.sum(falling)
            
        elif single_or_double == 'double':
            peak_indice_1 = peak_indices[0]
            peak_indice_2 = peak_indices[1]
            # nadir_indice_1 = nadir_indices[1]
            nadir_indice_1 = 0

            #NOTE: temporary solution to picking nadir; maybe better to get approx location of nadir through unsmoothed waveform
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

            average_ratio = (ratio_1 + ratio_2)/2

            return average_ratio

    #need to double check this method; not sure if i'm messing up the lengths for comparison (might need to +1 for some things?)
    #double check this
    def linearity(time, brf, single_or_double, rising_or_falling):
        nadir_clipping_length = 25
        peak_clipping_length = 125

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

            time_rising_1 = time[0+nadir_clipping_length:peak_indice_1-peak_clipping_length+1]
            rising_1 = brf[0+nadir_clipping_length:peak_indice_1-peak_clipping_length+1]

            time_falling_1 = time[peak_indice_1+peak_clipping_length:nadir_indice_1-nadir_clipping_length+1]
            falling_1 = brf[peak_indice_1+peak_clipping_length:nadir_indice_1-nadir_clipping_length+1]

            time_rising_2 = time[nadir_indice_1+nadir_clipping_length:peak_indice_2-peak_clipping_length+1]
            rising_2 = brf[nadir_indice_1+nadir_clipping_length:peak_indice_2-peak_clipping_length+1]

            time_falling_2 = time[peak_indice_2+peak_clipping_length:len(brf)-nadir_clipping_length]
            falling_2 = brf[peak_indice_2+peak_clipping_length:len(brf)-nadir_clipping_length]

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


    def slope(x, y, n):
        assert len(x) == len(y)
        numerator = n*np.sum(x*y) - np.sum(x)*np.sum(y)
        denominator = n*np.sum(x**2) - np.sum(x)**2
        return numerator/denominator

    #angle of inflection on peaks and nadir; angle of inflection doesn't really make too much sense for single cycled waveforms?
    def angle_of_inflection(time, brf, single_or_double, peak_or_nadir):
        line_length = 100
        
        peak_indices = signal.find_peaks(brf, distance = 750)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]

        # linear_line = np.arange(0, line_length, 1)

        #this only gives angle of peak, which isn't that useful for us?
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

            set_nadir_var(nadir_indice_1)

            new_time = np.linspace(0,2,len(time))

            time_pr1 = time[peak_indice_1-line_length+1:peak_indice_1+1]
            # time_pr1 = new_time[peak_indice_1-line_length+1:peak_indice_1+1]
            peak_rising_1 = brf[peak_indice_1-line_length+1:peak_indice_1+1]
            time_pf1 = time[peak_indice_1+1:peak_indice_1+line_length+1]
            # time_pf1 = new_time[peak_indice_1+1:peak_indice_1+line_length+1]
            peak_falling_1 = brf[peak_indice_1+1:peak_indice_1+line_length+1]

            time_pr2 = time[peak_indice_2-line_length+1:peak_indice_2+1]
            # time_pr2 = new_time[peak_indice_2-line_length+1:peak_indice_2+1]
            peak_rising_2 = brf[peak_indice_2-line_length+1:peak_indice_2+1]
            time_pf2 = time[peak_indice_2+1:peak_indice_2+line_length+1]
            # time_pf2 = new_time[peak_indice_2+1:peak_indice_2+line_length+1]
            peak_falling_2 = brf[peak_indice_2+1:peak_indice_2+line_length+1]

            time_nf1 = time[nadir_indice_1-line_length+1:nadir_indice_1+1]
            # time_nf1 = new_time[nadir_indice_1-line_length+1:nadir_indice_1+1]
            nadir_falling_1 = brf[nadir_indice_1-line_length+1:nadir_indice_1+1]
            time_nr1 = time[nadir_indice_1+1:nadir_indice_1+line_length+1]
            # time_nr1 = new_time[nadir_indice_1+1:nadir_indice_1+line_length+1]
            nadir_rising_1 = brf[nadir_indice_1+1:nadir_indice_1+line_length+1]

            peak_rising_slope_1 = brf_analysis.slope(time_pr1, peak_rising_1, line_length)
            peak_falling_slope_1 = brf_analysis.slope(time_pf1, peak_falling_1, line_length)

            peak_rising_slope_2 = brf_analysis.slope(time_pr2, peak_rising_2, line_length)
            peak_falling_slope_2 = brf_analysis.slope(time_pf2, peak_falling_2, line_length)
            
            nadir_falling_slope_1 = brf_analysis.slope(time_nf1, nadir_falling_1, line_length)
            nadir_rising_slope_1 = brf_analysis.slope(time_nr1, nadir_rising_1, line_length)

            linear_regressor = LinearRegression()
            # linear_regressor1 = LinearRegression()
            # linear_regressor2 = LinearRegression()
            linear_regressor.fit(time_nf1.reshape(-1,1), nadir_falling_1.reshape(-1,1))
            y_pred1 = linear_regressor.predict(time_nf1.reshape(-1,1))

            linear_regressor.fit(time_nr1.reshape(-1,1), nadir_rising_1.reshape(-1,1))
            y_pred2 = linear_regressor.predict(time_nr1.reshape(-1,1))
            
            x1 = np.arange(nadir_indice_1-line_length+1, nadir_indice_1+1, 1)
            x2 = np.arange(nadir_indice_1+1, nadir_indice_1+line_length+1, 1)
            set_x1(x1)
            set_y1(y_pred1)
            set_x2(x2)
            set_y2(y_pred2)
            
            # nadir_falling_slope_1 = brf_analysis.slope(x1, nadir_falling_1, line_length)
            # nadir_rising_slope_1 = brf_analysis.slope(x2, nadir_rising_1, line_length)

            # assert len(x1) == len(y_pred1)

            # falling = (y_pred1[len(y_pred1)-1] - y_pred1[0])/(time_nf1[len(time_nf1)-1] - time_nf1[0])
            # rising = (y_pred2[len(y_pred2)-1] - y_pred2[0])/(time_nr1[len(time_nr1)-1] - time_nr1[0])

            # print(falling)
            # print(rising)

            # set_falling_slope_var(nadir_falling_slope_1)
            # set_rising_slope_var(nadir_rising_slope_1)

            peak_angle_1 = math.atan(peak_rising_slope_1) - math.atan(peak_falling_slope_1)
            peak_angle_2 = math.atan(peak_rising_slope_2) - math.atan(peak_falling_slope_2)
            nadir_angle_1 = math.atan(nadir_falling_slope_1)+math.pi - math.atan(nadir_rising_slope_1)

            if peak_or_nadir == 'peak':
                return math.degrees((peak_angle_1+peak_angle_2)/2)
            elif peak_or_nadir == 'nadir':
                return math.degrees(nadir_angle_1)

    #for each bulb type, print out the stats for that particular concatenated waveform
    def for_type_print_stats(brf_database, single_or_double):
        bulb_types = database_processing.return_bulb_types(brf_database)
        for i in range(len(bulb_types)):
            print(bulb_types[i])
            new_database = database_processing.return_bulb_type_waveforms(brf_database,str(bulb_types[i]))
            same_type_list = database_processing.database_to_list(new_database)
            for item in same_type_list: #item: folder name; name; bulb type
                brf_list = brf_extraction(item[0], single_or_double).brf_list
                concatenated_brf = brf_extraction.concatenate_waveforms(brf_list)
                # print(brf_analysis.crest_factor(concatenated_brf))
                # print(brf_analysis.kurtosis(concatenated_brf))
                print(brf_analysis.skew(concatenated_brf))
            print()

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

    #brf is a column vector; len(brf) x 1 matrix
    def PCA(brf_list, num_components):
        row_mean = np.array([np.mean(brf_list, axis = 1)]).T

        A = brf_list - row_mean
        print(f'A Shape: {A.shape}')
        U,S,V_T = np.linalg.svd(A, full_matrices = False)
        V = V_T.T
        print(f'V Shape: {V.shape}')
        w = V[:,:num_components]
        print(f'w Shape: {w.shape}')
        pca_brf_list = w.T@brf_list.T

        return pca_brf_list, w

    #this method almost turned out just as ugly as the previous one...
    def feature_analysis(brf_database, method_name_list, single_or_double): #method_list vs method_name_list: 'angle_of_inflection' vs. 'Angle of Inflection'
        brf_database_list = database_processing.database_to_list(brf_database)
        
        '''
        ONE dataframe: (unique name)
            1) all statistics for EACH waveform of a BRF
                a) e.g. Eiko CFL 13W | CFL | Angle of Inflection | Integral Ratio | Crest Factor | Kurtosis | Skew | ... | etc.
                b) for the column names, find a way to dynamically add names to the dataframe
                c) create dataframe that only contains name and bulb type (so TWO columns initially)
                d) for loop to add columns to dataframe
        After dataframe is made:
            1) mean and variance of each waveform of a BRF for the WHOLE database
            2) all BRFs within a bulb type, filter by bulb type (using for loop as done below)
                a) of a bulb type, mean and variance of each waveform for bulb type database
                b) for a specific BRF (e.g. Eiko CFL 13W), get unique names for bulb type database
                    i) mean and variance for each BRF
                c) mean and variance of ALL waveforms within bulb type
        New dataframes:
            e.g. BRF Name | Bulb Type | Angle of Inflection (Mean) | Angle of Inflection (STD) | Integral Ratio (Mean) | Integral Ratio (Mean) | ... | etc.
            create general mean and variance list and add that list to an existing dataframe?? make this easy without having to do the crap above
            1) Mean and variance of each waveform (for a BRF) for the ENTIRE database
            2) Mean and variance of each waveform (for a BRF) for BULB TYPE database
            3) Mean and variance of each *BRF* for a BULB TYPE database
            4) Mean and variance of each BULB TYPE for the ENTIRE database
        Then, can do bar graphs with error bars in Excel or via matplotlib 
        '''

        #to add more methods/columns, go to export_all_to_csv method
        # df = database_processing.export_all_to_csv(brf_database, single_or_double)

        load_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\BRF Analysis\stat_analysis.csv'

        figure_save_directory = r'C:\Users\alexy\OneDrive\Documents\STIMA\BRF Analysis\Feature Analysis\Feature Analysis Figures'

        df = pd.read_csv(load_path)

        brf_names = df.BRF_Name.unique()
        bulb_types = df.Bulb_Type.unique() #only do first four bulb types

        #IS THERE A BETTER WAY TO DO THIS??
        #VALUES ARE NOT CLEARING???
        brf_name_list = list([])
        bulb_type_list = list([])
        integral_mean = list([])
        integral_CI = list([])
        angle_mean = list([])
        angle_CI = list([])
        crest_mean = list([])
        crest_CI = list([])
        kurtosis_mean = list([])
        kurtosis_CI = list([])
        skew_mean = list([])
        skew_CI = list([])

        #all waveforms in a BRF
        for name in brf_names:
            new_df = df.loc[df['BRF_Name'] == name]
            bulb_type = new_df['Bulb_Type'].tolist()[0]

            #make a general method that spits out new dataframe (??)
            mean_list, CI_list = database_processing.return_mean_CI_lists(new_df, method_name_list)

            assert len(mean_list) == len(CI_list)

            brf_name_list.append(name)
            bulb_type_list.append(bulb_type)
            integral_mean.append(mean_list[0])
            integral_CI.append(CI_list[0])
            angle_mean.append(mean_list[1])
            angle_CI.append(CI_list[1])
            crest_mean.append(mean_list[2])
            crest_CI.append(CI_list[2])
            kurtosis_mean.append(mean_list[3])
            kurtosis_CI.append(CI_list[3])
            skew_mean.append(mean_list[4])
            skew_CI.append(CI_list[4])

        #I think I can make this more automated with the method_name_list parameter...
        integral_save_path = figure_save_directory + '\\Comparison of BRFs in Entire Database\\integral_analysis_entire_database.png'
        angle_save_path = figure_save_directory + '\\Comparison of BRFs in Entire Database\\angle_analysis_entire_database.png'
        crest_factor_save_path = figure_save_directory + '\\Comparison of BRFs in Entire Database\\crest_factor_analysis_entire_database.png'
        kurtosis_save_path = figure_save_directory + '\\Comparison of BRFs in Entire Database\\kurtosis_analysis_entire_database.png'
        skew_save_path = figure_save_directory + '\\Comparison of BRFs in Entire Database\\skew_analysis_entire_database.png'

        string = 'for Entire Database'
        integral_title = 'Integral Ratio Analysis ' + string
        angle_title = 'Angle of Inflection Analysis' + string
        crest_factor_title = 'Crest Factor Analysis ' + string
        kurtosis_title = 'Kurtosis Analysis ' + string
        skew_title = 'Skew Analysis ' + string
        
        plots.feature_analysis_bar_graphs(brf_name_list, integral_mean, integral_CI, integral_title, integral_save_path)
        plots.feature_analysis_bar_graphs(brf_name_list, angle_mean, angle_CI, angle_title, angle_save_path)
        plots.feature_analysis_bar_graphs(brf_name_list, crest_mean, crest_CI, crest_factor_title, crest_factor_save_path)
        plots.feature_analysis_bar_graphs(brf_name_list, kurtosis_mean, kurtosis_CI, kurtosis_title, kurtosis_save_path)
        plots.feature_analysis_bar_graphs(brf_name_list, skew_mean, skew_CI, skew_title, skew_save_path)

        #CHANGE THIS TO BE DYNAMIC
        d = {'BRF Name': brf_name_list, 'Bulb Type': bulb_type_list, 'Integral Ratio (Mean)': integral_mean, 'Integral Ratio (95% CI)': integral_CI, 'Nadir Angle (Mean)': angle_mean, 'Nadir Angle (95% CI)': angle_CI, 'Crest Factor (Mean)': crest_mean, 'Crest Factor (95% CI)': crest_CI, 'Kurtosis (Mean)': kurtosis_mean, 'Kurtosis (95% CI)': kurtosis_CI, 'Skew (Mean)': skew_mean, 'Skew (95% CI)': skew_CI}

        stats_df = pd.DataFrame(data = d)
        file_name = 'mean_CI95_analysis_entire.csv'
        save_path = feature_analysis_save_directory + '\\' + file_name

        if os.path.exists(save_path):
            print('File exists: do you want to overwrite? (Y/N):')
            print(file_name)

            x = input()
            if x == 'Y':
                stats_df.to_csv(save_path)
        else:
            stats_df.to_csv(save_path)

        clear_lists(brf_name_list, bulb_type_list, integral_mean, integral_CI, angle_mean, angle_CI, crest_mean, crest_CI, kurtosis_mean, kurtosis_CI, skew_mean, skew_CI)

        #mean and variance for a bulb type (should only be 4 rows)
        for i in range(0, 4):
            bulb_type = bulb_types[i]
            new_df = df.loc[df['Bulb_Type'] == bulb_type]

            mean_list, CI_list = database_processing.return_mean_CI_lists(new_df, method_name_list)

            assert len(mean_list) == len(CI_list)

            bulb_type_list.append(bulb_type)
            integral_mean.append(mean_list[0])
            integral_CI.append(CI_list[0])
            angle_mean.append(mean_list[1])
            angle_CI.append(CI_list[1])
            crest_mean.append(mean_list[2])
            crest_CI.append(CI_list[2])
            kurtosis_mean.append(mean_list[3])
            kurtosis_CI.append(CI_list[3])
            skew_mean.append(mean_list[4])
            skew_CI.append(CI_list[4])

        integral_save_path = figure_save_directory + '\\Comparison of Bulb Types\\integral_analysis_bulb_types.png'
        angle_save_path = figure_save_directory + '\\Comparison of Bulb Types\\angle_analysis_bulb_types.png'
        crest_factor_save_path = figure_save_directory + '\\Comparison of Bulb Types\\crest_factor_analysis_bulb_types.png'
        kurtosis_save_path = figure_save_directory + '\\Comparison of Bulb Types\\kurtosis_analysis_bulb_types.png'
        skew_save_path = figure_save_directory + '\\Comparison of Bulb Types\\skew_analysis_bulb_types.png'

        string = 'for Bulb Types'
        integral_title = 'Integral Ratio Analysis ' + string
        angle_title = 'Angle of Inflection Analysis' + string
        crest_factor_title = 'Crest Factor Analysis ' + string
        kurtosis_title = 'Kurtosis Analysis ' + string
        skew_title = 'Skew Analysis ' + string

        plots.feature_analysis_bar_graphs(bulb_type_list, integral_mean, integral_CI, integral_title, integral_save_path)
        plots.feature_analysis_bar_graphs(bulb_type_list, angle_mean, angle_CI, angle_title, angle_save_path)
        plots.feature_analysis_bar_graphs(bulb_type_list, crest_mean, crest_CI, crest_factor_title, crest_factor_save_path)
        plots.feature_analysis_bar_graphs(bulb_type_list, kurtosis_mean, kurtosis_CI, kurtosis_title, kurtosis_save_path)
        plots.feature_analysis_bar_graphs(bulb_type_list, skew_mean, skew_CI, skew_title, skew_save_path)

        d = {'Bulb Type': bulb_type_list, 'Integral Ratio (Mean)': integral_mean, 'Integral Ratio (95% CI)': integral_CI, 'Nadir Angle (Mean)': angle_mean, 'Nadir Angle (95% CI)': angle_CI, 'Crest Factor (Mean)': crest_mean, 'Crest Factor (95% CI)': crest_CI, 'Kurtosis (Mean)': kurtosis_mean, 'Kurtosis (95% CI)': kurtosis_CI, 'Skew (Mean)': skew_mean, 'Skew (95% CI)': skew_CI}

        stats_df = pd.DataFrame(data = d)
        file_name = 'mean_CI95_analysis_bulb_type.csv'
        save_path = feature_analysis_save_directory + '\\' + file_name

        if os.path.exists(save_path):
            print('File exists: do you want to overwrite? (Y/N):')
            print(file_name)

            x = input()
            if x == 'Y':
                stats_df.to_csv(save_path)
        else:
            stats_df.to_csv(save_path)

        clear_lists(brf_name_list, bulb_type_list, integral_mean, integral_CI, angle_mean, angle_CI, crest_mean, crest_CI, kurtosis_mean, kurtosis_CI, skew_mean, skew_CI)

        #within a bulb type, compare BRFs
        for i in range(0,4):
            bulb_type = bulb_types[i]
            type_df = df.loc[df['Bulb_Type'] == bulb_type]

            brf_names_of_type = type_df.BRF_Name.unique()

            for name in brf_names_of_type:
                brf_of_type_df = type_df.loc[type_df['BRF_Name'] == name]

                mean_list, CI_list = database_processing.return_mean_CI_lists(brf_of_type_df, method_name_list)

                assert len(mean_list) == len(CI_list)

                brf_name_list.append(name)
                bulb_type_list.append(bulb_type)
                integral_mean.append(mean_list[0])
                integral_CI.append(CI_list[0])
                angle_mean.append(mean_list[1])
                angle_CI.append(CI_list[1])
                crest_mean.append(mean_list[2])
                crest_CI.append(CI_list[2])
                kurtosis_mean.append(mean_list[3])
                kurtosis_CI.append(CI_list[3])
                skew_mean.append(mean_list[4])
                skew_CI.append(CI_list[4])

            integral_save_path = figure_save_directory + '\\Comparison of BRFs of a Bulb Type\\' + bulb_type + '\\integral_analysis_bulb_types.png'
            angle_save_path = figure_save_directory + '\\Comparison of BRFs of a Bulb Type\\' + bulb_type + '\\angle_analysis_bulb_types.png'
            crest_factor_save_path = figure_save_directory + '\\Comparison of BRFs of a Bulb Type\\' + bulb_type + '\\crest_factor_analysis_bulb_types.png'
            kurtosis_save_path = figure_save_directory + '\\Comparison of BRFs of a Bulb Type\\' + bulb_type + '\\kurtosis_analysis_bulb_types.png'
            skew_save_path = figure_save_directory + '\\Comparison of BRFs of a Bulb Type\\' + bulb_type + '\\skew_analysis_bulb_types.png'

            string = 'for ' + bulb_type + ' Bulbs'
            integral_title = 'Integral Ratio Analysis ' + string
            angle_title = 'Angle of Inflection Analysis' + string
            crest_factor_title = 'Crest Factor Analysis ' + string
            kurtosis_title = 'Kurtosis Analysis ' + string
            skew_title = 'Skew Analysis ' + string
            
            plots.feature_analysis_bar_graphs(brf_names_of_type, integral_mean, integral_CI, integral_title, integral_save_path)
            plots.feature_analysis_bar_graphs(brf_names_of_type, angle_mean, angle_CI, angle_title, angle_save_path)
            plots.feature_analysis_bar_graphs(brf_names_of_type, crest_mean, crest_CI, crest_factor_title, crest_factor_save_path)
            plots.feature_analysis_bar_graphs(brf_names_of_type, kurtosis_mean, kurtosis_CI, kurtosis_title, kurtosis_save_path)
            plots.feature_analysis_bar_graphs(brf_names_of_type, skew_mean, skew_CI, skew_title, skew_save_path)

            d = {'BRF Name': brf_name_list, 'Bulb Type': bulb_type_list, 'Integral Ratio (Mean)': integral_mean, 'Integral Ratio (95% CI)': integral_CI, 'Nadir Angle (Mean)': angle_mean, 'Nadir Angle (95% CI)': angle_CI, 'Crest Factor (Mean)': crest_mean, 'Crest Factor (95% CI)': crest_CI, 'Kurtosis (Mean)': kurtosis_mean, 'Kurtosis (95% CI)': kurtosis_CI, 'Skew (Mean)': skew_mean, 'Skew (95% CI)': skew_CI}

            stats_df = pd.DataFrame(data = d)
            file_name = 'mean_CI95_analysis_' + bulb_type + '.csv'
            save_path = feature_analysis_save_directory + '\\' + file_name

            if os.path.exists(save_path):
                print('File exists: do you want to overwrite? (Y/N):')
                print(file_name)

                x = input()
                if x == 'Y':
                    stats_df.to_csv(save_path)
            else:
                stats_df.to_csv(save_path)

            clear_lists(brf_name_list, bulb_type_list, integral_mean, integral_CI, angle_mean, angle_CI, crest_mean, crest_CI, kurtosis_mean, kurtosis_CI, skew_mean, skew_CI)
            


class brf_classification():
    def compare_brfs(brf_database, num_comparisons, single_or_double): #num_comparisons is the number of comparisons we want to make (e.g. 3)
        bulb_types = database_processing.return_bulb_types(brf_database)
        brf_database_list = database_processing.database_to_list(brf_database)
        
        ground_list = list([])
        predicted_list = list([])

        bulb_type_list = database_processing.return_bulb_types(brf_database)

        #Personal note: if want to generate confusion matrix for bulb type, don't include outer bulb_type for loop
        #               otherwise, if want to compare unique BRFs, compare between the bulb type classification
        #               additionally, need to change what is being appended to the error list (type or name?)

        for bulb_type in bulb_type_list:
            print(bulb_type)
            same_type_database = database_processing.return_bulb_type_waveforms(brf_database,str(bulb_type))
            same_type_database_list = database_processing.database_to_list(same_type_database)
            same_type_name_list = database_processing.database_to_list(database_processing.return_names(same_type_database))
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

    def my_distance(weights):
        print(weights)
        return weights

    #classification_type is for either the BRF name or BRF type; options are either 'name' or 'type'
    def train_KNN(brf_database, n, classification_type, num_test_waveforms, single_or_double):
        number_neighbors = n
        brf_database_list = database_processing.database_to_list(brf_database)
        
        KNN_input = np.ones((5))
        KNN_output = list([])

        crest_factor_array = np.array([])
        kurtosis_array = np.array([])
        skew_array = np.array([])

        #for each element:
        #   index 0: [crest factor, kurtosis, skew]
        #   index 1: BRF name
        crest_factor_prediction = np.array([])
        kurtosis_prediction = np.array([])
        skew_prediction = np.array([])
        # brf_name_output_label = list([])
        KNN_prediction_list = list([])

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
                smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[i], savgol_window), mov_avg_w_size)
                # linearity = brf_analysis.linearity(time_list[i], smoothed, single_or_double, 'falling')
                angle = brf_analysis.angle_of_inflection(time_list[i], smoothed, single_or_double, 'nadir')
                integral_ratio = brf_analysis.integral_ratio(smoothed, single_or_double)
                crest_factor = brf_analysis.crest_factor(waveform_list[i])
                kurtosis = brf_analysis.kurtosis(waveform_list[i])
                skew = brf_analysis.skew(waveform_list[i])
                # input_param = [linearity, angle, integral_ratio, crest_factor, kurtosis, skew]
                input_param = np.array([angle, integral_ratio, crest_factor, kurtosis, skew])
                # input_param = [crest_factor, kurtosis, skew]
                if i < num_test_waveforms: #determines number of test/training waveforms
                    # crest_factor_prediction = np.append(crest_factor_prediction, crest_factor)
                    # kurtosis_prediction = np.append(kurtosis_prediction, kurtosis)
                    # skew_prediction = np.append(skew_prediction, skew)

                    if classification_type == 'name':
                        KNN_prediction_list.append([input_param, brf_name])
                    elif classification_type == 'type':
                        KNN_prediction_list.append([input_param, bulb_type])
                    # brf_name_output_label.append(brf_name)
                    # brf_name_output_label.append(bulb_type)
                else:
                    # crest_factor_array = np.append(crest_factor_array, crest_factor)
                    # kurtosis_array = np.append(kurtosis_array, kurtosis)
                    # skew_array = np.append(skew_array, skew)

                    KNN_input = np.vstack((KNN_input, input_param))
                    if classification_type == 'name':
                        KNN_output.append(brf_name)
                    if classification_type == 'type':
                        KNN_output.append(bulb_type)
            print(f'{brf_name} Finished') #this is just to make sure the program is running properly

        '''
        # crest_factor_array_normalized = raw_waveform_processing.normalize(crest_factor_array)
        # kurtosis_array_normalized = raw_waveform_processing.normalize(kurtosis_array)
        # skew_array_normalized = raw_waveform_processing.normalize(skew_array)

        # crest_factor_prediction_normalized = raw_waveform_processing.normalize(crest_factor_prediction)
        # kurtosis_prediction_normalized = raw_waveform_processing.normalize(kurtosis_prediction)
        # skew_prediction_normalized = raw_waveform_processing.normalize(skew_prediction)

        # KNN_prediction_values = np.vstack((crest_factor_prediction_normalized, kurtosis_prediction_normalized, skew_prediction_normalized)).T
        # KNN_prediction_list = list(zip(KNN_prediction_values, brf_name_output_label))
        # KNN_input = np.vstack((crest_factor_array_normalized, kurtosis_array_normalized, skew_array_normalized)).T
        '''

        brf_KNN_model = KNeighborsClassifier(n_neighbors = number_neighbors) #used 9 because about 9 waveforms for each BRF
        # brf_KNN_model = KNeighborsClassifier(n_neighbors = number_neighbors, weights = 'distance') #used 9 because about 9 waveforms for each BRF

        KNN_input = KNN_input[1:len(KNN_input)]

        brf_KNN_model.fit(KNN_input, KNN_output)

        return brf_KNN_model, KNN_prediction_list

    def KNN(brf_database, number_neighbors, classification_type, num_test_waveforms, single_or_double, Tallied): #True or False for tallied
        # brf_database = database_processing.drop_bulb_type_column(brf_database)
        # database_as_list = database_processing.database_to_list(brf_database)

        no_match = True

        bulb_type_list = database_processing.return_bulb_types(brf_database)
        brf_recall_list = list([])

        #requires and input list and output list to train the model
        for bulb_type in bulb_type_list:
            print(bulb_type)
            same_type_database = database_processing.return_bulb_type_waveforms(brf_database,str(bulb_type))
            same_type_name_list = database_processing.database_to_list(same_type_database.Name)
            # same_type_database = database_processing.drop_bulb_type_column(same_type_database)

            brf_KNN_model, KNN_prediction_list = brf_classification.train_KNN(same_type_database, number_neighbors, classification_type, num_test_waveforms, single_or_double)

        # brf_KNN_model, KNN_prediction_list = brf_classification.train_KNN(brf_database, number_neighbors, classification_type)

            ground_list = list([])
            predicted_list = list([])

            true_positive = 0
            false_neg = 0
            total = len(KNN_prediction_list)

            tallied_matrix = np.zeros((len(same_type_name_list), len(same_type_name_list)))
            total_matrix = np.zeros((len(same_type_name_list), len(same_type_name_list)))

            for prediction in KNN_prediction_list:
                input_data = prediction[0]
                expected_output = prediction[1]
                output = brf_KNN_model.predict([input_data])[0]

                probabilities = brf_KNN_model.predict_proba([input_data])[0]
                row_total = np.full(probabilities.shape, number_neighbors)

                neighbor_indices = brf_KNN_model.kneighbors([input_data])[1][0]

                index = same_type_name_list.index(expected_output)

                # print(f'Expected: {expected_output}')
                # print(f'Output: {output}')

                print(f'{expected_output}; Index = {index}')
                for model_index in neighbor_indices:
                    same_type_name_list_index = brf_KNN_model._y[model_index]
                    print(same_type_name_list_index)
                    if same_type_name_list[same_type_name_list_index] == expected_output:
                        true_positive += 1
                        ground_list.append(expected_output)
                        predicted_list.append(expected_output)
                        no_match = False
                        break
                if no_match:
                    if expected_output == output:
                        print(f'Expected: {expected_output}')
                        print(f'Output: {output}')
                        false_neg += 1
                    ground_list.append(expected_output)
                    predicted_list.append(output)

                no_match = True

                print()

                # if expected_output == output:
                #     print(f'Expected: {expected_output}; Output: {output}')
                #     true_positive += 1

                if Tallied:

                    probabilities = brf_KNN_model.predict_proba([input_data])
                    row_total = np.full(probabilities.shape,number_neighbors)

                    expected_index = same_type_name_list.index(expected_output)

                    #creating matrixs of same shape as tallied_matrix and total_matrix to add matricies
                    temp_probabilities_matrix = np.zeros(probabilities.shape)
                    temp_total_matrix = np.zeros(probabilities.shape)
                    zeros_row = np.zeros(probabilities.shape)
                    for i in range(len(tallied_matrix)):
                        if i == expected_index:
                            temp_probabilities_matrix = np.vstack((temp_probabilities_matrix,probabilities))
                            temp_total_matrix = np.vstack((temp_total_matrix,row_total))
                        else:
                            temp_probabilities_matrix = np.vstack((temp_probabilities_matrix,zeros_row))
                            temp_total_matrix = np.vstack((temp_total_matrix,zeros_row))

                    #removing first row

                    temp_probabilities_matrix = temp_probabilities_matrix[1:len(temp_probabilities_matrix)]*number_neighbors
                    temp_total_matrix = temp_total_matrix[1:len(temp_total_matrix)]

                    tallied_matrix += temp_probabilities_matrix
                    total_matrix += temp_total_matrix

                    brf_KNN_model = None

            print(f'True Positive: {true_positive}')
            print(f'False Negative: {false_neg}')
            print(total)

            precision = true_positive/total
            print(f'Precision: {precision}')
            print()

            if Tallied:
                tallied_precision = np.trace(tallied_matrix)/np.trace(total_matrix)
                plots.KNN_confusion_matrix_tallies(tallied_matrix, total_matrix, same_type_name_list, f'[angle, integral_ratio, crest_factor, kurtosis, skew]\nTallied Confusion Matrix for {bulb_type} Bulb Type Using KNN \n(~7 double cycles for training, 3 for testing)\nOverall Correctness: {tallied_precision}')

            plots.confusion_matrix_unique(ground_list, predicted_list, same_type_name_list, f'[angle, integral_ratio, crest_factor, kurtosis, skew]\nConfusion Matrix for {bulb_type} Bulb Type Using KNN\nClosest {num_test_waveforms} Neighbors\n(~7 double cycles for training, 3 for testing)\nOverall Correctness: {precision}')

    #I AM REWRITING CODE AGAIN; FIX LATER FOR A GENERAL KNN MODEL
    def PCA_KNN(brf_database, single_or_double, num_training, num_components, num_neighbors): #num_training ==> number of waveforms for training
        bulb_types = database_processing.return_bulb_types(brf_database)
        brf_database_list = database_processing.database_to_list(brf_database)

        bulb_type_list = database_processing.return_bulb_types(brf_database)

        for bulb_type in bulb_type_list:
            print(bulb_type)
            same_type_database = database_processing.return_bulb_type_waveforms(brf_database,str(bulb_type))
            same_type_database_list = database_processing.database_to_list(same_type_database)
            same_type_name_list = database_processing.database_to_list(database_processing.return_names(same_type_database))

            brf_database_list = same_type_database_list
            
            length_list = np.array([])
            
            training_list = []
            training_label_list = []

            test_list = []
            test_label_list = []

            ground_list = list([])
            predicted_list = list([])

            for i in range(len(brf_database_list)):
                #right now, going to truncate all waveforms, but need to figure out how to interpolate later; maybe it doesn't matter because there's a lot of points?

                folder_name = brf_database_list[i][0]
                brf_name = brf_database_list[i][1]
                bulb_type = brf_database_list[i][2]
                extracted_lists = brf_extraction(folder_name, single_or_double)
                time_list = extracted_lists.time_list
                waveform_list = extracted_lists.brf_list

                print(brf_name)

                for j in range(len(waveform_list)):
                    if j < num_training:
                        length_list = np.hstack((length_list, len(waveform_list[j])))
                        training_list.append(waveform_list[j])
                        training_label_list.append(brf_name)
                    else:
                        length_list = np.hstack((length_list, len(waveform_list[j])))
                        test_list.append(waveform_list[j])
                        test_label_list.append(brf_name)

            min_length = int(np.amin(length_list))
            max_length = int(np.amax(length_list))

            mean_length = np.mean(length_list)
            std_length = math.sqrt(np.sum((length_list-mean_length)**2/(len(length_list)-1)))

            training_data = np.ones((min_length))
            test_data = np.ones((min_length))

            print(f'Min length: {min_length}')
            print(f'Max length: {max_length}')
            print(f'Mean length: {mean_length}')
            print(f'STD: {std_length}')

            #FIGURE OUT SOME WAY FOR INTERPOLATION

            for j in range(len(training_list)):
                training_data = np.vstack((training_data, training_list[j][:min_length]))

            for j in range(len(test_list)):
                test_data = np.vstack((test_data, test_list[j][:min_length]))

            training_data = training_data[1:len(training_data)]
            test_data = test_data[1:len(test_data)]

            pca_training_data, w = brf_analysis.PCA(training_data, num_components)

            pca_test_data = w.T@test_data.T

            pca_training_data = pca_training_data.T
            pca_test_data = pca_test_data.T

            KNN_classifier = KNeighborsClassifier(n_neighbors = num_neighbors)
            KNN_classifier.fit(pca_training_data, training_label_list)

            true_positive = 0
            total = len(pca_test_data)

            tallied_matrix = np.zeros((len(same_type_name_list), len(same_type_name_list)))
            total_matrix = np.zeros((len(same_type_name_list), len(same_type_name_list)))

            for j in range(len(pca_test_data)):
                input_data = pca_test_data[j]
                expected_output = test_label_list[j]
                output = KNN_classifier.predict([input_data])[0]

                ground_list.append(expected_output)
                predicted_list.append(output)

                if expected_output == output:
                    # print(f'Expected: {expected_output}; Output: {output}')
                    true_positive += 1

                probabilities = KNN_classifier.predict_proba([input_data])
                row_total = np.full(probabilities.shape,num_neighbors)

                expected_index = same_type_name_list.index(expected_output)

                #creating matrixs of same shape as tallied_matrix and total_matrix to add matricies
                temp_probabilities_matrix = np.zeros(probabilities.shape)
                temp_total_matrix = np.zeros(probabilities.shape)
                zeros_row = np.zeros(probabilities.shape)
                for i in range(len(tallied_matrix)):
                    if i == expected_index:
                        temp_probabilities_matrix = np.vstack((temp_probabilities_matrix,probabilities))
                        temp_total_matrix = np.vstack((temp_total_matrix,row_total))
                    else:
                        temp_probabilities_matrix = np.vstack((temp_probabilities_matrix,zeros_row))
                        temp_total_matrix = np.vstack((temp_total_matrix,zeros_row))

                #removing first row
                temp_probabilities_matrix = temp_probabilities_matrix[1:len(temp_probabilities_matrix)]*num_neighbors
                temp_total_matrix = temp_total_matrix[1:len(temp_total_matrix)]

                tallied_matrix += temp_probabilities_matrix
                total_matrix += temp_total_matrix

            tallied_precision = np.trace(tallied_matrix)/np.trace(total_matrix)

            precision = true_positive/total
            print(f'Precision: {precision}')
            print()

            plots.KNN_confusion_matrix_tallies(tallied_matrix, total_matrix, same_type_name_list, f'Tallied Confusion Matrix for {bulb_type} Bulb Type Using PCA KNN with {num_components} Features \n(~7 double cycles for training, 3 for testing)\nOverall Correctness: {tallied_precision}')
            plots.confusion_matrix_unique(ground_list, predicted_list, same_type_name_list, f'Confusion Matrix for {bulb_type} Bulb Type Using PCA KNN with {num_components} Features \n(~7 double cycles for training, 3 for testing)\nOverall Correctness: {precision}')


if __name__ == "__main__":
    method_name_list = ['Integral_Ratio', 'Nadir_Angle', 'Crest_Factor', 'Kurtosis', 'Skew']
    brf_database = database_processing(database_path).brf_database

    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'halco_led_10w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_6p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_8w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_7p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_12w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_5p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_11w'].index)

    # brf_classification.compare_brfs(brf_database, 3, 'double')
    # brf_classification.KNN(brf_database, 3, 'name', 3, 'double', Tallied = False)
    # brf_analysis.brf_gradient_analysis(brf_database, 'double', gradient_save_path)

    # brf_analysis.test_analysis_method(brf_database, 'integral_ratio', 'double')

    # brf_analysis.test_analysis_method(brf_database, 'test', 'single')
    # brf_classification.PCA_KNN(brf_database, 'double', 7, 10, 3)

    # database_processing.export_all_to_csv(brf_database, method_name_list, 'double')

    brf_analysis.feature_analysis(brf_database, method_name_list, 'double')