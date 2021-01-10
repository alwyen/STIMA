import time
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import math
import os
import seaborn as sn

database_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\bulb_database_master.csv'
base_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files'
savgol_window = 31

brf_analysis_save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\BRF Analysis'

class plots():
    def show_plot(waveform):
        plt.plot(waveform)
        plt.show()
    
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

    def clean_brf(brf):
        nadir_indices = signal.find_peaks(-brf, distance = 750)[0]
        cleaned_brf = brf[nadir_indices[0]:nadir_indices[2]] #first and third nadir
        return cleaned_brf

    def normalize(array):
        min = np.amin(array)
        max = np.amax(array)
        normalized = (array[:]-min)/(max-min)
        return normalized

    def savgol(brf, savgol_window):
        smoothed = ss.savgol_filter(brf, savgol_window, 3)
        return smoothed

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
            brf_list = []
            path = base_path + '\\' + brf_folder_name
            os.chdir(path)
            num_files = (len([name for name in os.listdir(path) if os.path.isfile(name)]))
            os.chdir(cwd)
            for i in range(num_files):
                brf_path = path + '\\waveform_' + str(i) + '.csv'
                brf = raw_waveform_processing(brf_path).brf
                brf = raw_waveform_processing.normalize(raw_waveform_processing.clean_brf(brf))
                nadir_indices = signal.find_peaks(-brf, distance = 750)[0]
                brf1 = brf[0:nadir_indices[1]]
                brf2 = brf[nadir_indices[1]:len(brf)]
                if single_or_double == 'double':
                    brf_list.append(brf)
                elif single_or_double == 'single':
                    brf_list.append(brf1)
                    brf_list.append(brf2)
            self.brf_list = brf_list

        def concatenate_waveforms(waveform_list):
            waveform = np.array([])
            for i in range(len(waveform_list)):
                waveform = np.concatenate((waveform, waveform_list[i]),0)
            return waveform

#brf_analysis class contains all the statistical tests/analysis methods
class brf_analysis():
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

    #for each bulb type, print out the stats for that particular concatenated waveform
    def for_type_print_stats(brf_database, single_or_double):
        bulb_types = database_processing.return_bulb_types(brf_database) #drop bulb type column later?
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

    def gradient(data):
        horizontal_gradient = np.array([[1, 0, -1]])
        gradient_x = signal.convolve(data, horizontal_gradient, mode = 'valid')
        plots.show_plot(gradient_x)
        return gradient_x

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
        
        KNN_input = list([])
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
            waveform_list = brf_extraction(folder_name, single_or_double).brf_list
            #ignoring the first waveform – will use that for classification
            for i in range(len(waveform_list)):
                crest_factor = brf_analysis.crest_factor(waveform_list[i])
                kurtosis = brf_analysis.kurtosis(waveform_list[i])
                skew = brf_analysis.skew(waveform_list[i])
                input_param = [crest_factor, kurtosis, skew]
                if i < num_test_waveforms: #determines number of test/training waveforms
                    crest_factor_prediction = np.append(crest_factor_prediction, crest_factor)
                    kurtosis_prediction = np.append(kurtosis_prediction, kurtosis)
                    skew_prediction = np.append(skew_prediction, skew)

                    if classification_type == 'name':
                        KNN_prediction_list.append([input_param, brf_name])
                    elif classification_type == 'type':
                        KNN_prediction_list.append([input_param, bulb_type])
                    # brf_name_output_label.append(brf_name)
                    # brf_name_output_label.append(bulb_type)
                else:
                    crest_factor_array = np.append(crest_factor_array, crest_factor)
                    kurtosis_array = np.append(kurtosis_array, kurtosis)
                    skew_array = np.append(skew_array, skew)
                    KNN_input.append(input_param)
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

        brf_KNN_model.fit(KNN_input, KNN_output)

        return brf_KNN_model, KNN_prediction_list

    def KNN(brf_database, number_neighbors, classification_type, num_test_waveforms, single_or_double):
        # brf_database = database_processing.drop_bulb_type_column(brf_database)
        # database_as_list = database_processing.database_to_list(brf_database)

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
            total = len(KNN_prediction_list)

            tallied_matrix = np.zeros((len(same_type_name_list), len(same_type_name_list)))
            total_matrix = np.zeros((len(same_type_name_list), len(same_type_name_list)))

            for prediction in KNN_prediction_list:
                input_data = prediction[0]
                expected_output = prediction[1]
                output = brf_KNN_model.predict([input_data])[0]

                ground_list.append(expected_output)
                predicted_list.append(output)

                if expected_output == output:
                    print(f'Expected: {expected_output}; Output: {output}')
                    true_positive += 1

                probabilities = brf_KNN_model.predict_proba([input_data])
                row_total = np.full(probabilities.shape,number_neighbors)

                expected_index = same_type_name_list.index(expected_output)

                #creating matrixs of same shape as tallied_matrix and tota_matrix to add matricies
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

                print(tallied_matrix)

            precision = true_positive/total
            print(f'Precision: {precision}')
            print()

            plots.KNN_confusion_matrix_tallies(tallied_matrix, total_matrix, same_type_name_list, f'Tallied Confusion Matrix for {bulb_type} Bulb Type Using KNN \n(~7 double cycles for training, 3 for testing)')
            plots.confusion_matrix_unique(ground_list, predicted_list, same_type_name_list, f'Confusion Matrix for {bulb_type} Bulb Type Using KNN \n(~7 double cycles for training, 3 for testing)')


if __name__ == "__main__":
    brf_database = database_processing(database_path).brf_database

    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'halco_led_10w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_6p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'philips_led_8w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_7p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'sylvania_led_12w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_5p5w'].index)
    brf_database = brf_database.drop(brf_database[brf_database['Folder_Name'] == 'westinghouse_led_11w'].index)

    brf_classification.compare_brfs(brf_database, 3, 'double')
    # brf_classification.KNN(brf_database, 7, 'name', 3, 'double')