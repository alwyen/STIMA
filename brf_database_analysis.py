import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

database_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\bulb_database_master.csv'

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

class waveform_data():
    def __init__(self, brf_path):
        brf_data = np.genfromtxt(csv_path, delimiter = ',')
        brf_data = brf_data[1:len(brf_data)] #removing first row
        time, brf, voltage = np.hsplit(brf_data, len(brf_data[0]))
        self.time = np.hstack(time)
        self.brf = np.hstack(brf)
        self.voltage = np.hstack(voltage)

class brf_extraction():
    
    base_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files'

    def __init__(self, brf_path):

        brf_list = []
        path = base_path + '\\' + base_path + '\\' + brf_path
        num_files = (len([name for name in os.listdir('.') if os.path.isfile(name)]))
        for i in range(num_files):
            brf_path = path + '\\waveform_' + str(i) + '.csv'
            brf = waveform_data(brf_path).brf
            brf_list.append(brf) #double check this?



class brf_analysis():
    def crest_factor(brf):
        peak_value = np.amax(brf)
        rms = math.sqrt(np.sum(np.array(brf))/len(brf))
        crest_factor = peak_value/rms
        return crest_factor

if __name__ == "__main__":
    brf = brf_extraction(base_path)
    # database = brf_database(database_path).brf_database
    # bulb_types = brf_database.return_bulb_types(database)
    # for i in range(len(bulb_types)):
    #     print(bulb_types[i])
    #     new_database = brf_database.return_bulb_type_waveforms(database,str(bulb_types[i]))
    #     brf_list = database_to_list(new_database)
