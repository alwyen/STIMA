import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt
import math
import os
import glob

import pdb

# local file imports
import define
from plotting import plots
from data_processing import database_processing, brf_extraction, scope, ACam

#CLEAN UP TIME
#fix all the global methods/variable names
#remove data_processing class methods that aren't being used; just look up pandas methods dude
#make export_all_to_csv dynamic; maybe the solution is to hand in a list of features to be analyzed? somehow need to call the method name then; associate "method name" with "name"?
#check if "train_KNN" and "KNN" can be simplified

#for skipping plotting in KNN
skip = False

def skip_plots(val):
    global skip
    skip = val

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
                        error_score = scope.sheinin_min_error(brf_1, brf_2)
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

    #name_k is for bulb identification, which I am not doing at the moment! --> (1/23/22) is that for the three closest neighbors...?
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

            print(input_data)
            print(expected_output)
            print(brf_name)
            print()

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

        # TEMPORARY
        # TEMPORARY
        # TEMPORARY
        return brf_KNN_model


    #get different KNN_in, KNN_out, KNN_prediction_list, and then RUN THE METHOD
    def KNN_analysis_pkl(pkl_path, brf_database, classification_type, weights, Entire, num_test_waveforms, num_features, number_neighbors, name_k):
        # other parameters
        ########################################################################################################################################################
        MisClass = False            #no misclassification analysis done with this method, so set MisClass to False
        ########################################################################################################################################################

        if Entire:
            KNN_in, KNN_out, KNN_prediction_list = brf_classification.KNN_in_out_pkl(pkl_path, None, classification_type, num_test_waveforms, num_features, k_fold_CV=False, split_number=0)

            # TEMPORARY
            # TEMPORARY
            # TEMPORARY
            brf_KNN_model = brf_classification.KNN(brf_database, KNN_in, KNN_out, KNN_prediction_list, number_neighbors, classification_type, num_test_waveforms, num_features, weights, Entire, name_k, MisClass)
            return brf_KNN_model
        else:
            bulb_type_list = database_processing.return_bulb_types(brf_database)
            for bulb_type in bulb_type_list:
                print(bulb_type)
                same_type_database = database_processing.same_type_df(brf_database,str(bulb_type))
                KNN_in, KNN_out, KNN_prediction_list = brf_classification.KNN_in_out_pkl(pkl_path, bulb_type, classification_type, num_test_waveforms, num_features, k_fold_CV=False, split_number=0)
                brf_classification.KNN(same_type_database, KNN_in, KNN_out, KNN_prediction_list, number_neighbors, classification_type, num_test_waveforms, num_features, weights, Entire, MisClass)
                print()

    '''
    DEFINITION: takes in previously trained BRF model; classification is for bulb TYPE

    INPUT:      brf_KNN_model       KNN model
                csv_path            key-value database of ACam traces; this contains master csv file to access folders
                                    format should be:
                                        relative folder path | Bulb type
                                    
                                    relative folder path should contain csv files of traces; use glob to access traces
                                        run stats like pkl_KNN_in_out; note that ACam traces are SINGLE cycle
                folder_name         name of folder to create path and access files
                debugging           flag to enable more debugging features

    OUPUT:      nothing
    '''
    def classify_ACam_BRFs(brf_KNN_model, ACam_path, folder_name, reconstruct, classification_type, binary_classification, average_only, debugging):
        csv_path = os.path.join(ACam_path, folder_name)
        
        cwd = os.getcwd()
        os.chdir(csv_path)
        file_list = glob.glob('*.csv')

        KNN_input = list()
        KNN_output_type = list([])

        num_traces = len(file_list)
        tru_pos_counter = 0
        bin_tru_pos_counter = 0
        type_correct = False
        bin_correct = False

        # creating empty array of length of entries in CSV file
        temp_waveform = abs(np.array(pd.read_csv(file_list[0])['Intensity']) - 1)
        new_temp_waveform = ACam.reconstruct_LIVE_ACam_waveform(temp_waveform)

        if reconstruct:
            new_temp_waveform = ACam.reconstruct_LIVE_ACam_waveform(temp_waveform)
        else:
            new_temp_waveform = temp_waveform
        sum_traces = np.zeros(len(new_temp_waveform))

        for csv_file in file_list:
            # print(csv_file)
            df = pd.read_csv(csv_file)

            # need this step because somehow the waveforms are flipped
            norm_waveform = abs(np.array(df['Intensity']) - 1)
            if reconstruct:
                # new_waveform = brf_analysis.reconstruct_ACam_waveform(norm_waveform)
                new_waveform = ACam.reconstruct_LIVE_ACam_waveform(norm_waveform)
            else:
                new_waveform = norm_waveform

            sum_traces += new_waveform

            if average_only:
                continue

            input_param = ACam.extract_features_ACam(new_waveform)
            output = brf_KNN_model.predict([input_param])[0]
            print(output)
            print()

            # TEMPORARY CODE; this needs to be fixed for binary classification
            if output == classification_type:
                tru_pos_counter += 1
                type_correct = True
            else:
                type_correct = False

            if binary_classification == 'IH':
                if output == 'Incandescent' or output == 'Halogen':
                    bin_tru_pos_counter += 1
                    bin_correct = True
                else:
                    bin_correct = False

            elif binary_classification == 'CL':
                if output == 'CFL' or output == 'LED':
                    bin_tru_pos_counter += 1
                    bin_correct = True
                else:
                    bin_correct = False

            if debugging:
                print(csv_file)
                plt.plot(new_waveform)
                plt.xlabel('Sample Number')
                plt.ylabel('Normalized Intensity')
                plt.show()
                # quit()

        #     # turn off error plotting
        #     if not debugging:
        #         if type_correct:
        #             if g_once:
        #                 plt.plot(new_waveform, color='g', label='Type Correct')
        #                 g_once = False
        #             else:
        #                 plt.plot(new_waveform, color='g')
        #         elif bin_correct:
        #             if y_once:
        #                 plt.plot(new_waveform, color='y', label='Binary Correct')
        #                 y_once = False
        #             else:
        #                 plt.plot(new_waveform, color='y')
        #         else:
        #             if r_once:
        #                 plt.plot(new_waveform, color='r', label='Incorrect')
        #                 r_once = False
        #             else:
        #                 plt.plot(new_waveform, color='r')
        
        # if not debugging:
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     label_order = ['Type Correct', 'Binary Correct', 'Incorrect']
        #     # label_order = ['Type Correct', 'Binary Correct']

        #     order_list = [0, 0, 0]
        #     # order_list = [0, 0]

        #     for i in range(len(label_order)):
        #         order_list[i] = labels.index(label_order[i])

        #     print(order_list)

        #     # # pass handle & labels lists along with order as below
        #     plt.legend([handles[i] for i in order_list], [labels[i] for i in order_list])
        #     plt.xlabel('Sample Number', fontsize=18)
        #     plt.xticks(np.arange(0, len(new_waveform)+1, step=20), fontsize=18)
        #     plt.ylabel('Normalized Intensity', fontsize=18)
        #     plt.yticks(np.arange(0, 1.1, step=0.25), fontsize=18)
        #     plt.tight_layout()
        #     plt.show()
                

            # if output == 'LED':
            #     print(csv_file)
                # print()

        # avg_traces = sum_traces/num_traces
        # plt.plot(avg_traces)
        # plt.show()

        # avg_traces_input_param = brf_analysis.extract_features_ACam(avg_traces)
        # avg_traces_output = brf_KNN_model.predict([avg_traces_input_param])[0]
        # print(f'Result of averaging traces: {avg_traces_output}')

        if not average_only:
            accuracy = tru_pos_counter / num_traces * 100
            bin_accuracy = bin_tru_pos_counter / num_traces * 100
            print(f'Number of traces: {num_traces}')
            print(f'Number classified correctly: {tru_pos_counter}')
            print(f'Type classification accuracy: {accuracy}%')
            print(f'Number of correct binary classification: {bin_tru_pos_counter}')
            print(f'Binary classification accuracy: {bin_accuracy}%')
        os.chdir(cwd)
        # return KNN_input


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
    # pkl save path
    csv_pkl_save_path = os.path.join(define.STIMA_dir, 'bulb_database')
    smoothed_save_path = os.path.join(define.STIMA_dir, 'bulb_database', 'smoothed_brf_examples')

    # pkl file path
    pkl_path = os.path.join(define.STIMA_dir, 'bulb_database', 'KNN_downsized.pkl')
    # pkl_path = os.path.join(STIMA_dir, 'bulb_database', 'KNN_downsized_orig.pkl') # this is from July 2021

    # brf_database instatiation
    brf_database = database_processing(define.database_path).brf_database
    brf_database = database_processing.filtering(brf_database)

    # weights
    #'Entire' is for the entire database
    # weights = np.array([0.25, 0.0, 0.75, 0.5, 0.75, 1.0, 1.0])
    # weights = np.array([0.0, 0.75, 0.5, 0.75, 1.0, 1.0])
    # weights = np.array([0.75, 0.5, 0.75, 1.0, 1.0])
    # weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) # --> integral average, peak location, crest, kurtosis, skew
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # --> integral ratio, integral average, peak location, crest, kurtosis, skew
    # weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    ############################################################################################
    ############################################################################################
    ############################################################################################

    # database_processing.ACam_renaming(define.file_renaming_path, define.consolidated_folder_path)
    # quit()

    # reconstructed the waveforms to start at nadir and end at nadir
    # folder_name = 'CFL_1'
    # folder_name = 'ACam_Training_Data/Philips_Halogen_39W'
    # folder_name = 'ACam_Training_Data/GE_Incandescent_40W'
    # folder_name = 'Incan_Indoor_Test_3'

    ACam_train = False
    # folder_name = 'LIVE/Halogen_Indoor_Test_1'
    # folder_name = 'LIVE/LIVE_Mixed_Outdoor_Test_7_25_22/LIVE_Mixed_Outdoor_Test_7/CFL'
    folder_name = 'LIVE/Outdoor_Testing/Halogen'
    reconstruct = True
    type = 'Halogen'
    bin_class = 'IH'    # "IH" for incandescent/halogen
                        # "CL" for CFL or LED
    average_only = False
    debugging = False

    # pickles features
    print('Run preprocessing? [y]/other')
    val = input()
    if val == 'y':
        # database_processing.pkl_KNN_in_out(brf_database, 'double', csv_pkl_save_path, num_features=6)
        database_processing.pkl_KNN_in_out(brf_database, define.ACam_db_path, 'double', csv_pkl_save_path, num_features=6, ACam_train=ACam_train)

    # runs KNN analysis on pkl file
    # TEMPORARILY RETURNING MODEL
    brf_KNN_model = brf_classification.KNN_analysis_pkl(pkl_path, brf_database, 'type', weights, Entire = True, num_test_waveforms=3, number_neighbors=4, num_features=6, name_k=3)
    #after choosing K, train on train+validation set; final with test set

    quit()

    brf_classification.classify_ACam_BRFs(brf_KNN_model, define.ACam_path, folder_name, reconstruct=reconstruct, classification_type=type, binary_classification=bin_class, average_only=average_only, debugging=debugging)

    folder_name_0 = 'LIVE/Outdoor_Testing/Halogen'
    folder_name_1 = 'LIVE/Outdoor_Testing/Incandescent'

    # have this method return the BRFs that were misclassified as LEDs
    halogen_LED_list, incan_LED_list = plots.ACam_plot_similar_BRFs(brf_KNN_model, define.ACam_path, folder_name_0, folder_name_1)

    scope_LED_path = os.path.join(define.STIMA_dir, 'bulb_database', 'incan_halogen_LED')
    plots.compare_incan_halogen_LED_plots(halogen_LED_list, incan_LED_list, scope_LED_path)

    # k-fold analysis
    # brf_classification.k_fold_analysis(pkl_path, brf_database, 'type', weights, num_test_waveforms = 999, num_features = 5, min_num_neighbors = 2, max_num_neighbors = 10, num_splits = 12, MisClass = True)

    ############################################################################################################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################

    # #plotting
    # '''
    # smoothed_brf_path = os.path.join(STIMA_dir, 'bulb_database', 'smoothed_brf_examples', 'CFL')
    # plots.plot_three_waveforms(smoothed_brf_path, smoothed_save_path, 'CFL')

    # smoothed_brf_path = os.path.join(STIMA_dir, 'bulb_database', 'smoothed_brf_examples', 'Halogen')
    # plots.plot_three_waveforms(smoothed_brf_path, smoothed_save_path, 'Halogen')

    # smoothed_brf_path = os.path.join(STIMA_dir, 'bulb_database', 'smoothed_brf_examples', 'Incandescent')
    # plots.plot_three_waveforms(smoothed_brf_path, smoothed_save_path, 'Incandescent')

    # smoothed_brf_path = os.path.join(STIMA_dir, 'bulb_database', 'smoothed_brf_examples', 'LED')
    # plots.plot_three_waveforms(smoothed_brf_path, smoothed_save_path, 'LED')

    # smoothed_brf_path = r'C:\Users\alexy\Dropbox\STIMA\bulb_database\smoothed_brf_examples\GRFP'
    # plots.plot_three_waveforms(smoothed_brf_path, smoothed_save_path, 'GRFP')
    # '''

    # # example plots
    # '''
    # csv_path = r'/Users/alexyen/Dropbox/STIMA/bulb_database/brf_database/csv_files/eiko_cfl_13w/waveform_0.csv'
    # waveform = raw_waveform_processing(csv_path)
    # plots.show_plot(waveform.voltage)
    # '''