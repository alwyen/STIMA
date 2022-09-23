import numpy as np
from scipy import signal
import os
import glob
import define

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
                integral_ratio = brf_analysis.integral_ratio(smoothed, single_or_double)

                # these values are not downsized; we're not even using the smoothed values for our analyses
                # integral_ratio = brf_analysis.integral_ratio(waveform_list[i], single_or_double)
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
                    new_waveform = brf_analysis.reconstruct_LIVE_ACam_waveform(norm_waveform)
                    ACam_input_param = brf_analysis.extract_features_ACam(new_waveform)
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