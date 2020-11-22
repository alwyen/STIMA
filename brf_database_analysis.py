import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

database_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\bulb_database_master.csv'

class brf_database():
    def __init__(self, database_path):
        whole_database = pd.read_csv(database_path)
        self.brf_database = whole_database.loc[:,['Folder_Name', 'Name', 'Bulb_Type']]

    def return_bulb_types(brf_database):
        return brf_database.Bulb_Type.unique()

    def return_bulb_type_waveforms(brf_database, bulb_type):
        return brf_database.loc[brf_database['Bulb_Type'] == bulb_type]

if __name__ == "__main__":
    database = brf_database(database_path).brf_database
    bulb_types = brf_database.return_bulb_types(database)
    for i in range(len(bulb_types)):
        print(bulb_types[i])
        brf_database.return_bulb_type_waveforms(database,str(bulb_types[i]))