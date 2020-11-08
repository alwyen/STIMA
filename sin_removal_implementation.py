import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.fftpack import fft, ifft
from natsort import natsorted, ns


def dft_idft(signal):
    boundary = 50

    sample_rate = 1005 #sampling frequency
    num_samples = len(signal)
    # print(num_samples)
    sample_spacing = 1/sample_rate
    # print(sample_spacing)
    xf = np.linspace(0, 1.0/(2*sample_spacing), num_samples//2)
    # print(xf)
    orig_fft = fft(signal)
    yf = 2.0/num_samples*np.abs(orig_fft[0:num_samples//2])
    peak_indices = ss.find_peaks(yf)[0]
    xf_values = xf[peak_indices]
    yf_values = yf[peak_indices]
    yf_values[yf_values < 0.5] = 0
    xf_values[yf_values == 0] = 0
    orig_fft[10] = 0
    orig_fft[15] = 0

    print(xf_values[:10])
    print(yf_values[:10])
    # yf[yf == yf_values] = 0
    # yf[10] = 0
    # yf[15] = 0

    # print(yf)
    plt.plot(xf[:boundary],yf[:boundary])
    plt.show()

    return ifft(orig_fft)

def bandpass_filter(signal):
    sos = ss.butter(10, 20, 'bandpass', fs=1005, output='sos')
    filtered = ss.sosfilt(sos, signal)
    return filtered

if __name__ == "__main__":

    t = np.linspace(0,1,1000, False)
    sin_10hz = np.sin(2*np.pi*10*t)
    sin_15hz = np.sin(2*np.pi*15*t)
    sin_25hz = np.sin(2*np.pi*25*t)
    noise = np.random.random_sample((len(t),))

    clean_signal = sin_10hz + sin_15hz + sin_25hz
    signal = sin_10hz + sin_15hz + sin_25hz + noise
    # sinusoids_removed = dft_idft(signal)
    # second = dft_idft(sinusoids_removed)
    filtered = bandpass_filter(signal)

    fig, ax = plt.subplots(4,1)
    fig.tight_layout(pad = 2.0)

    ax[0].plot(noise)
    ax[0].set_title('Noise')

    ax[1].plot(clean_signal)
    ax[1].set_title('Clean 10Hz + 15Hz Signal')

    ax[2].plot(signal)
    ax[2].set_title('Clean Signal + Noise')

    ax[3].plot(filtered)
    ax[3].set_title('Signal with 10Hz and 15Hz Frequency Components Removed')

    plt.show()
