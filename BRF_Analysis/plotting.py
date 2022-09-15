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

'''
TEMPORARY
TEMPORARY
TEMPORARY
'''

#brf_analysis class contains all the statistical tests/analysis methods
class brf_analysis():
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
        integral_ratio = brf_analysis.ACam_integral_ratio(norm_waveform)
        int_avg = brf_analysis.ACam_cycle_integral_avg(norm_waveform)
        peak_loc = brf_analysis.ACam_peak_location(norm_waveform)
        
        crest_factor = brf_analysis.crest_factor(norm_waveform)
        kurtosis = brf_analysis.kurtosis(norm_waveform)
        skew = brf_analysis.skew(norm_waveform)

        input_param = np.array([integral_ratio, int_avg, peak_loc, crest_factor, kurtosis, skew])
        return input_param

    #gets distance squared of two BRFs; not euclidean distance (did not sqrt the result)
    #this is the method used in Sheinin's paper
    def sheinin_min_error(brf_1, brf_2):
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
        smoothed_gradient = raw_waveform_processing.savgol(gradient_x, define.savgol_window)
        # plots.save_gradient_plot(data, smoothed_gradient, name)
        plots.save_gradient_plot(data, gradient_x, name)
        return gradient_x

    #enforces that the integral (or sum) is equal to 1 (edit: huh...??)
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

            # ratio_1 = np.sum(rising_1)/np.sum(falling_1)
            # ratio_2 = np.sum(rising_2)/np.sum(falling_2)
            ratio_1 = (np.sum(rising_1)/len(rising_1)) / (np.sum(falling_1)/len(falling_1))
            ratio_2 = (np.sum(rising_2)/len(rising_2)) / (np.sum(falling_2)/len(falling_2))

            average_ratio = (ratio_1 + ratio_2)/2

            return average_ratio

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
                return (cc_1[0][1] + cc_3[0][1])/2 #return the average of rising correlations

            elif rising_or_falling == 'falling':
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
                avg_cc = (cc_1[0][1] + cc_3[0][1])/2

                return avg_cc #return the average of rising correlations

            elif rising_or_falling == 'falling':
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

            # nadir_indice_1 is the end of the first cycle
            ratio_1 = peak_indice_1/nadir_indice_1
            ratio_2 = (peak_indice_2-nadir_indice_1)/(len(brf) - nadir_indice_1)
            
            return (ratio_1 + ratio_2)/2

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

    def ACam_cycle_integral_avg(brf):
        int_avg = np.sum(brf) / len(brf)
        return int_avg

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
            brf_analysis.gradient(averaged_brf, name_list[i])
            print(f'{name_list[i]} + done')

        os.chdir(cwd)

'''
TEMPORARY
TEMPORARY
TEMPORARY
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