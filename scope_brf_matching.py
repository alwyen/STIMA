import numpy as np
import matplotlib.pyplot as plt
eiko_cfl_13w = r'C:\Users\alexy\OneDrive\Documents\STIMA\bulb_database\csv_files\eiko_cfl_13w'
eiko_cfl_13w_0_path = eiko_cfl_13w + '\\' + 'waveform_0.csv' #figure out some automated way for this
#is there a fast/efficient way to load all the data so that I can just pick and choose without having to
#hardcode path in?

def show_plot(brf_data):
    plt.plot(brf_data)
    plt.show()

#implement some filtering/window averaging?
class scope_brf():
    def __init__(self, csv_path):
        brf_data = np.genfromtxt(eiko_cfl_13w_0_path, delimiter = ',')
        brf_data = brf_data[1:len(brf_data)] #removing first row
        time, brf, voltage = np.hsplit(brf_data, len(brf_data[0]))
        self.time = time
        self.brf = brf
        self.voltage = voltage

if __name__ == "__main__":
    eiko_cfl_13w_0 = scope_brf(eiko_cfl_13w_0_path) #waveform_0
    show_plot(eiko_cfl_13w_0.brf)