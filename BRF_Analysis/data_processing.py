import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import signal, stats
import os
import glob

import define
from feature_extraction import scope

'''
this class does all the processing on the database side from the master CSV file; df has the following columns:
    Folder_Name
    Name
    Bulb_Type

Example: [eiko_cfl_13w | Eiko CFL 13W | CFL]

Init: database folder path; might not work for ACam data

'''
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
    def pkl_KNN_in_out(brf_database, ACam_db_path, single_or_double, save_path, num_features, ACam_train):
        brf_database_list = database_processing.database_to_list(brf_database)
        
        KNN_input = list()
        KNN_output_type = list([])
        KNN_output_name = list([])

        row_list = list()

        max_val = -9999
        min_val = 9999
        down_list = list()

        # making these variables for assertions
        scope_input_param = None

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

                '''
                # smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(downsampled_BRF, savgol_window), mov_avg_w_size)
                # # linearity = brf_analysis.linearity(time_list[i], smoothed, single_or_double, 'falling')
                # angle = brf_analysis.angle_of_inflection(time_list[i], smoothed, single_or_double, 'nadir')
                # integral_ratio = brf_analysis.integral_ratio(smoothed, single_or_double)
                # int_avg = brf_analysis.cycle_integral_avg(downsampled_BRF, single_or_double)
                # peak_loc = brf_analysis.peak_location(downsampled_BRF, single_or_double)
                # crest_factor = brf_analysis.crest_factor(downsampled_BRF)
                # kurtosis = brf_analysis.kurtosis(downsampled_BRF)
                # skew = brf_analysis.skew(downsampled_BRF)
                '''

                smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[i], define.savgol_window), define.mov_avg_w_size)
                # linearity = brf_analysis.linearity(time_list[i], smoothed, single_or_double, 'falling')
                # angle = brf_analysis.angle_of_inflection(time_list[i], smoothed, single_or_double, 'nadir')
                integral_ratio = scope.integral_ratio(smoothed, single_or_double)

                # these values are not downsized; we're not even using the smoothed values for our analyses
                # integral_ratio = brf_analysis.integral_ratio(waveform_list[i], single_or_double)
                int_avg = scope.cycle_integral_avg(waveform_list[i], single_or_double)
                peak_loc = scope.peak_location(waveform_list[i], single_or_double)
                crest_factor = scope.crest_factor(waveform_list[i])
                kurtosis = scope.kurtosis(waveform_list[i])
                skew = scope.skew(waveform_list[i])

                # int_avg = brf_analysis.cycle_integral_avg(smoothed, single_or_double)                
                # peak_loc = brf_analysis.peak_location(smoothed, single_or_double)
                # crest_factor = brf_analysis.crest_factor(smoothed)
                # kurtosis = brf_analysis.kurtosis(smoothed)
                # skew = brf_analysis.skew(smoothed)

                # define the number of parameters we want in our KNN model
                # input_param = [linearity, angle, integral_ratio, crest_factor, kurtosis, skew]
                input_param = np.array([integral_ratio, int_avg, peak_loc, crest_factor, kurtosis, skew])
                # input_param = np.array([int_avg, peak_loc, crest_factor, kurtosis, skew])
                
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

                # downsampling is for making sure that each waveform has the same number of data points (i.e. 1600)
                downsample_amount = len(waveform_list[i]) - len(downsampled_BRF)
                if downsample_amount < min_val:
                    min_val = downsample_amount
                if downsample_amount > max_val:
                    max_val = downsample_amount
                down_list.append(downsample_amount)

                # store just for assertion later
                scope_input_param = input_param
                # plots.show_plot(waveform_list[i])
                # plots.show_plot(downsampled_BRF)

            print(f'{brf_name} Finished') #this is just to make sure the program is running properly

        if ACam_train:
            # adding ACam data to training
            print()
            print('ACam data now:')
            cwd = os.getcwd()
            os.chdir(ACam_db_path)
            ACam_db_df = pd.read_csv(os.path.join(ACam_db_path, 'ACam_db.csv'))
            folder_list = ACam_db_df['folder_name']
            name_list = ACam_db_df['brf_name']
            type_list = ACam_db_df['brf_type']

            for i in range(len(folder_list)):
                folder_path = os.path.join(ACam_db_path, folder_list[i])
                name = name_list[i]
                type = type_list[i]

                os.chdir(folder_path)
                file_list = glob.glob('*.csv')

                for csv_file in file_list:
                    # print(csv_file)
                    df = pd.read_csv(csv_file)
                    norm_waveform = abs(np.array(df['Intensity']) - 1)
                    new_waveform = ACam.reconstruct_LIVE_ACam_waveform(norm_waveform)
                    ACam_input_param = ACam.extract_features_ACam(new_waveform)
                    assert len(scope_input_param) == len(ACam_input_param)

                    KNN_input.append(ACam_input_param)
                    KNN_output_type.append(type)
                    KNN_output_name.append(name)

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

    def filtering(brf_database):
        # removing BRFs that induce errors
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'halco_led_10w'].index)
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_6p5w'].index)
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_8w'].index)
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_7p5w'].index)
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_12w'].index)
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_5p5w'].index)
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_11w'].index)

        #taking out these BRFs increased accuracy; labels are different
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'halco_halogenxenon_39w'].index)
        brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_halogenincandescent_65w'].index)

        return brf_database

    '''
    this should be in a preprocessing file

    DESCRIPTION: renames files (e.g. ACam_0.csv, ACam_1.csv, etc.) into a the `consolidated_folder_path` directory
                 (files are in multiple directories)

    INPUT:      file_renaming_path      path of the all the consolidated files
                    ex. file_renaming
    '''

    def ACam_renaming(file_renaming_path, consolidated_folder_path):
        cwd = os.getcwd()

        # example folder: LIVE_Mixed_Outdoor_Test_0
        os.chdir(file_renaming_path)
        test_folders = glob.glob('*')

        # example: CFL, Halogen, Incandescent, LED
        os.chdir(consolidated_folder_path)
        type_folders = glob.glob('*')

        min_count = 0
        max_count = 0

        for type in type_folders:

            # ex. Outdoor_Testing/CFL
            type_folders_path = os.path.join(consolidated_folder_path, type)
            # print(type_folders_path)

            for folder in test_folders: # folder => LIVE_Mixed_Outdoor_Test_0
                data_path = os.path.join(file_renaming_path, folder)
                # print(data_path)

                os.chdir(data_path)
                type_data_path = glob.glob('*') # should spit out the folder types
                
                for type_data_folder in type_data_path:
                    if type == type_data_folder:
                        
                        # ex: LIVE_Mixed_Outdoor_Test_0/CFL/
                        ACam_data_path = os.path.join(data_path, type_data_folder)
                        # print(ACam_data_path)
                        # quit()

                        os.chdir(ACam_data_path)

                        # ex. ACam_0.csv, ACam_1.csv, etc...
                        ACam_list = glob.glob('*.csv')

                        max_count += len(ACam_list)
                        for file_name in ACam_list:

                            new_file_name = os.path.join(type_folders_path, 'ACam_' + str(min_count) + '.csv')

                            os.rename(file_name, new_file_name)
                            min_count += 1
                    min_count = max_count
            min_count = 0
            max_count = 0

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


# feature extraction
# feature extraction
# feature extraction

class general():
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
        smoothed_gradient = raw_waveform_processing.savgol(gradient_x, define.savgol_window)
        # plots.save_gradient_plot(data, smoothed_gradient, name)
        # plots.save_gradient_plot(data, gradient_x, name)
        return gradient_x

    #enforces that the integral (or sum) is equal to 1 (edit: huh...??)
    def normalize_cycle(brf):
        integral = np.sum(brf)
        normalized = brf/integral
        return normalized

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

            # nadir_indice_1 is the end of the first cycle
            ratio_1 = peak_indice_1/nadir_indice_1
            ratio_2 = (peak_indice_2-nadir_indice_1)/(len(brf) - nadir_indice_1)
            
            return (ratio_1 + ratio_2)/2

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
            smoothed_brf = raw_waveform_processing.savgol(comparison_list[i], define.savgol_window)
            averaged_brf = raw_waveform_processing.moving_average(smoothed_brf, define.mov_avg_w_size)
            general.gradient(averaged_brf, name_list[i])
            print(f'{name_list[i]} + done')

        os.chdir(cwd)

class scope():
    #gets distance squared of two BRFs; not euclidean distance (did not sqrt the result)
    #this is the method used in Sheinin's paper
    def scope_sheinin_min_error(brf_1, brf_2):
        brf_1, brf_2 = raw_waveform_processing.truncate_longer(brf_1, brf_2)
        error = np.sum(np.square(np.absolute(np.array(brf_1) - np.array(brf_2))))
        return error

    #this ratio is defined by the left side of the peak/right side of the peak
    #(rising/falling)
    def scope_integral_ratio(brf, single_or_double):
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

            integral_normalized_1 = general.normalize_cycle(brf[0:nadir_indice_1])
            integral_normalized_2 = general.normalize_cycle(brf[nadir_indice_1+1:len(brf)])

            new_brf = np.hstack((integral_normalized_1, integral_normalized_2))

            rising_1 = new_brf[0:peak_indice_1+1]
            falling_1 = new_brf[peak_indice_1:nadir_indice_1+1]
            rising_2 = new_brf[nadir_indice_1:peak_indice_2+1]
            falling_2 = new_brf[peak_indice_2:len(brf)]

            # ratio_1 = np.sum(rising_1)/np.sum(falling_1)
            # ratio_2 = np.sum(rising_2)/np.sum(falling_2)
            ratio_1 = (np.sum(rising_1)/len(rising_1)) / (np.sum(falling_1)/len(falling_1))
            ratio_2 = (np.sum(rising_2)/len(rising_2)) / (np.sum(falling_2)/len(falling_2))

            average_ratio = (ratio_1 + ratio_2)/2

            return average_ratio

    # get the slope between two points; not sure which class this method should be in yet
    def slope(x, y, n):
        assert len(x) == len(y)
        numerator = n*np.sum(x*y) - np.sum(x)*np.sum(y)
        denominator = n*np.sum(x**2) - np.sum(x)**2
        #https://stackoverflow.com/questions/27784528/numpy-division-with-runtimewarning-invalid-value-encountered-in-double-scalars
        return numerator/denominator

    #angle of inflection on peaks and nadir; angle of inflection doesn't really make too much sense for single cycled waveforms?
    #the angle is between two cycles
    def scope_angle_of_inflection(time, brf, single_or_double, peak_or_nadir):
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
            peak_rising_slope_1 = scope.slope(time_pr1, peak_rising_1, line_length)
            peak_falling_slope_1 = scope.slope(time_pf1, peak_falling_1, line_length)

            #slopes for second peak
            peak_rising_slope_2 = scope.slope(time_pr2, peak_rising_2, line_length)
            peak_falling_slope_2 = scope.slope(time_pf2, peak_falling_2, line_length)
            
            #slopes for nadir
            nadir_falling_slope_1 = scope.slope(time_nf1, nadir_falling_1, line_length)
            nadir_rising_slope_1 = scope.slope(time_nr1, nadir_rising_1, line_length)

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

            #calculating angles for peaks and nadir
            peak_angle_1 = math.atan(peak_rising_slope_1) - math.atan(peak_falling_slope_1)
            peak_angle_2 = math.atan(peak_rising_slope_2) - math.atan(peak_falling_slope_2)
            nadir_angle_1 = math.atan(nadir_falling_slope_1)+math.pi - math.atan(nadir_rising_slope_1)

            if peak_or_nadir == 'peak':
                return math.degrees((peak_angle_1+peak_angle_2)/2)
            elif peak_or_nadir == 'nadir':
                return math.degrees(nadir_angle_1)
                
    #cycle integral (so just the sum of the cycle), but need to divide by the cycle length to account of differences in cycle length
    #(integral/cycle length)
    def scope_cycle_integral_avg(brf, single_or_double):
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

class ACam():
    def ACam_integral_ratio(brf):
        peak_indices = signal.find_peaks(brf)[0]

        # instantiation
        peak_index = None

        peak_index_max = 0

        for index in peak_indices:
            if brf[index] > peak_index_max:
                peak_index = index
                peak_index_max = brf[index]

        rising = brf[0:peak_index+1]
        falling = brf[peak_index:len(brf)]

        ratio = (np.sum(rising)/len(rising))/(np.sum(falling)/len(falling))
        return ratio

    def reconstruct_ACam_waveform(norm_waveform):
        peak_indices = signal.find_peaks(norm_waveform)[0]
        nadir_indices = signal.find_peaks(-norm_waveform)[0]

        # instantiation
        peak_index = None
        nadir_index = None

        peak_index_max = 0
        nadir_index_min = 1

        for index in peak_indices:
            if norm_waveform[index] > peak_index_max:
                peak_index = index
                peak_index_max = norm_waveform[index]

        for index in nadir_indices:
            if norm_waveform[index] < nadir_index_min:
                nadir_index = index
                nadir_index_min = norm_waveform[index]

        if peak_index < nadir_index:
            falling = norm_waveform[peak_index:nadir_index]
            rising_1 = norm_waveform[:peak_index]
            rising_2 = norm_waveform[nadir_index-1:]

            # rising_2 --> rising_1 --> falling
            reconstructed_waveform = np.concatenate((rising_2, rising_1, falling), axis=0)
            return reconstructed_waveform

        elif nadir_index < peak_index:
            rising = norm_waveform[nadir_index:peak_index]
            falling_1 = norm_waveform[:nadir_index]
            falling_2 = norm_waveform[peak_index-1:]

            # rising --> falling_2 --> falling_1

            reconstructed_waveform = np.concatenate((rising, falling_2, falling_1), axis=0)
            return reconstructed_waveform

    def reconstruct_LIVE_ACam_waveform(norm_waveform):
        peak_indices = signal.find_peaks(norm_waveform)[0]
        nadir_indices = signal.find_peaks(-norm_waveform)[0]

        # instantiation
        peak_index = None
        nadir_index = None

        peak_index_max = 0
        nadir_index_min = 1

        for index in peak_indices:
            if norm_waveform[index] > peak_index_max:
                peak_index = index
                peak_index_max = norm_waveform[index]

        for index in nadir_indices:
            if norm_waveform[index] < nadir_index_min:
                nadir_index = index
                nadir_index_min = norm_waveform[index]

        reconstructed_waveform = np.concatenate((norm_waveform[nadir_index:], norm_waveform[:nadir_index]), axis=0)
        return reconstructed_waveform

    def extract_features_ACam(norm_waveform):
        # EVERYTHING IS W.R.T. THE NADIR
        integral_ratio = ACam.ACam_integral_ratio(norm_waveform)
        int_avg = ACam.ACam_cycle_integral_avg(norm_waveform)
        peak_loc = ACam.ACam_peak_location(norm_waveform)
        
        crest_factor = general.crest_factor(norm_waveform)
        kurtosis = general.kurtosis(norm_waveform)
        skew = general.skew(norm_waveform)

        input_param = np.array([integral_ratio, int_avg, peak_loc, crest_factor, kurtosis, skew])
        return input_param

    # peak location divided by the length of the waveform/BRF
    # this is W.R.T. the NADIR
    def ACam_peak_location(brf):
        peak_indices = signal.find_peaks(brf)[0]

        # instantiation
        peak_index = None

        peak_index_max = 0

        for index in peak_indices:
            if brf[index] > peak_index_max:
                peak_index = index
                peak_index_max = brf[index]

        ratio = peak_index/len(brf)
        return ratio
    
    def ACam_cycle_integral_avg(brf):
        int_avg = np.sum(brf) / len(brf)
        return int_avg