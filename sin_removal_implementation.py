import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.fftpack import fft, ifft
from natsort import natsorted, ns


def dft_idft(signal):
    boundary = 50

    sample_rate = 2010 #sampling frequency
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

    # yf[yf == yf_values] = 0
    # yf[10] = 0
    # yf[15] = 0

    print(yf)
    plt.plot(xf[:boundary],yf[:boundary])
    plt.show()

    return ifft(orig_fft)

if __name__ == "__main__":

    t = np.linspace(0,1,1000, False)
    sin_10hz = np.sin(2*np.pi*10*t)
    sin_15hz = np.sin(2*np.pi*15*t)
    noise = np.random.random_sample((len(t),))

    clean_signal = sin_10hz + sin_15hz
    signal = sin_10hz + sin_15hz + noise
    sinusoids_removed = dft_idft(signal)
    print('again')
    second = dft_idft(sinusoids_removed)


    fig, ax = plt.subplots(4,1)
    fig.tight_layout(pad = 2.0)

    ax[0].plot(noise)

    ax[1].plot(clean_signal)

    ax[2].plot(signal)

    ax[3].plot(sinusoids_removed)

    plt.show()
