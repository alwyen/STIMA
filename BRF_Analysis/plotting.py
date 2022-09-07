import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy import signal

import define
from scope_processing import raw_waveform_processing, brf_extraction

'''
think I need to create a preprocessing file first for this file?
'''

# all plotting graphs
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

    '''
    plotting three different waveforms for Figure 1 in BuildSys paper
    '''
    def plot_three_waveforms(csv_brf_path, save_path, title_name):
        cwd = os.getcwd()
        os.chdir(csv_brf_path)
        csv_names = glob.glob('*.csv')
        colors = list(['r', 'g', 'b'])
        temp_labels = list(['Incandescent', 'CFL', 'LED'])

        # font_size = 18
        plt.rcParams["font.family"] = "Times New Roman"
        font_size = 11

        # fig, axs = plt.subplots(len(csv_names), constrained_layout=True, figsize = (8,15))
        for i in range(len(csv_names)):
            processed = raw_waveform_processing(csv_names[i])
            time = processed.time
            brf = processed.brf
            # time, brf = raw_waveform_processing.clean_brf(time, brf)
            brf = raw_waveform_processing.normalize(brf)
            smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(brf, define.savgol_window), define.mov_avg_w_size)
            smoothed_time = raw_waveform_processing.moving_average(time, define.mov_avg_w_size)
            # plt.plot(time, brf, colors[i])
            plt.plot(time, brf, colors[i], linewidth = 3, alpha = 1 - 0.25*i)
            # plt.plot(smoothed_time, smoothed, colors[i], linewidth = 3)
            # plt.plot(smoothed_time, smoothed, colors[i], linewidth = 3, markersize = 1, label = temp_labels[i])
            # plt.plot(normalized_smoothed, colors[i])
            plt.xlabel('Time (s)', fontsize = font_size)
            plt.ylabel('Normalized Intensity', fontsize = font_size)
            plt.xticks(fontsize = font_size)
            plt.yticks(fontsize = font_size)
            plt.locator_params(axis='x', nbins=5)
            # plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            # axs[i].plot(time, brf)
            # axs[i].set_xlabel('Time (s)', fontsize = 18)
            # axs[i].set_ylabel('Normalized Intensity', fontsize = 18)

        # plt.title(title_name, fontsize = 18)
        plt.legend(loc = 'upper left', fontsize = font_size, prop={'size': 9})
        plt.tight_layout()
        plt.show()
        # plt.savefig(save_path + '\\' + title_name + '.png')
        # plt.clf()

        os.chdir(cwd)

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

    '''
    not using this anymore
    '''
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

    '''
    this is a very specific plot; another plot for the BuildSys paper

    Description: plotting misclassification similarities between two types of light bulbs

    folder_name_0 should be for `halogen`
    folder_name_1 should be for 'incandescent'

    '''
    def ACam_plot_similar_BRFs(brf_KNN_model, ACam_path, folder_name_0, folder_name_1):
        csv_path_0 = os.path.join(ACam_path, folder_name_0)
        csv_path_1 = os.path.join(ACam_path, folder_name_1)
        
        cwd = os.getcwd()

        os.chdir(csv_path_0)
        file_list_0 = glob.glob('*.csv')

        h_once = True
        i_once = True

        halogen_LED_list = list()
        incan_LED_list = list()

        for csv_file in file_list_0:
            # print(csv_file)
            df = pd.read_csv(csv_file)

            # need this step because somehow the waveforms are flipped
            norm_waveform = abs(np.array(df['Intensity']) - 1)
            new_waveform = brf_analysis.reconstruct_LIVE_ACam_waveform(norm_waveform)

            input_param = brf_analysis.extract_features_ACam(new_waveform)
            output = brf_KNN_model.predict([input_param])[0]

            if output != 'Halogen':
                if output == 'Incandescent':
                    if h_once:
                        plt.plot(new_waveform, color='b', label='Halogen', alpha=0.5)
                        h_once = False
                    else:
                        plt.plot(new_waveform, color='b', alpha=0.5)
                if output == 'LED':
                    halogen_LED_list.append(new_waveform)


        os.chdir(csv_path_1)
        file_list_1 = glob.glob('*.csv')

        for csv_file in file_list_1:
            # print(csv_file)
            df = pd.read_csv(csv_file)

            # need this step because somehow the waveforms are flipped
            norm_waveform = abs(np.array(df['Intensity']) - 1)
            new_waveform = brf_analysis.reconstruct_LIVE_ACam_waveform(norm_waveform)

            input_param = brf_analysis.extract_features_ACam(new_waveform)
            output = brf_KNN_model.predict([input_param])[0]

            if output != 'Incandescent':
                if output == 'Halogen':
                    if i_once:
                        plt.plot(new_waveform, color='orange', label='Incandescent', alpha=0.5)
                        i_once = False
                    else:
                        plt.plot(new_waveform, color='orange', alpha=0.5)
                if output == 'LED':
                    incan_LED_list.append(new_waveform)

        font_size = 13

        plt.legend(fontsize=font_size, loc='lower center')
        plt.xlabel('Sample Number', fontsize=font_size)
        plt.xticks(np.arange(0, len(new_waveform)+1, step=20), fontsize=font_size)
        plt.ylabel('Normalized Intensity', fontsize=font_size)
        plt.yticks(np.arange(0, 1.1, step=0.25), fontsize=font_size)
        plt.tight_layout()
        plt.show()
        
        os.chdir(cwd)

        return halogen_LED_list, incan_LED_list

    '''
    plotting similarities between halogen, incandescent, and LED BRFs for buildsys paper
    '''
    def compare_incan_halogen_LED_plots(halogen_LED_list, incan_LED_list, LED_path):
        h_once = True
        i_once = True
        l_once = True

        cwd = os.getcwd()
        os.chdir(LED_path)
        led_folder_list = glob.glob('*')
        
        for brf in halogen_LED_list:
            if h_once:
                plt.plot(brf, color='b', label='Halogen', alpha=0.5)
                h_once = False
            else:
                plt.plot(brf, color='b', alpha=0.5)

        for brf in incan_LED_list:
            if i_once:
                plt.plot(brf, color='orange', label='Incandescent', alpha=0.5)
                i_once = False
            else:
                plt.plot(brf, color='orange', alpha=0.5)

        for led_folder in led_folder_list:
            waveform_list = brf_extraction(led_folder, 'double').brf_list
            smoothed = raw_waveform_processing.moving_average(raw_waveform_processing.savgol(waveform_list[1], define.savgol_window), define.mov_avg_w_size)
            nadir_indices = signal.find_peaks(-smoothed, distance = 750)[0]
            single_cycle = smoothed[:nadir_indices[0]]
            downsampled_single_cycle = raw_waveform_processing.normalize(signal.resample(single_cycle, 80))
            if l_once:
                plt.plot(downsampled_single_cycle, color='g', label='LED', alpha = 0.5)
                l_once = False
            else:
                plt.plot(downsampled_single_cycle, color='g', alpha = 0.5)
        
        font_size = 13
        plt.legend(fontsize=font_size, loc='lower center')
        plt.xlabel('Sample Number', fontsize=font_size)
        plt.xticks(np.arange(0, len(brf)+1, step=20), fontsize=font_size)
        plt.ylabel('Normalized Intensity', fontsize=font_size)
        plt.yticks(np.arange(0, 1.1, step=0.25), fontsize=font_size)
        plt.tight_layout()
        plt.show()
        # for brf in halogen_LED_list:
        #     plt.plot(brf)