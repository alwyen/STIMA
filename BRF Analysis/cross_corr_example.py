from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def show_plot(array):
    plt.plot(array)
    plt.show()

def normalize(array):
    normalized_array = []
    min = np.amin(array)
    max = np.amax(array)
    #FIND A BETTER WAY TO DO THIS (USING NP ARRAY)
    for i in range(len(array)):
        normalized_array.append((array[i] - min)/(max - min))
    normalized_array = np.array(normalized_array)
    return normalized_array

x = np.ones(128)
# x = x + np.random.randn(len(x))
# x = normalize(x)
# show_plot(x)
y = np.ones(128)
y = y + np.random.rand(len(y))
y = y/2 + 0.25
# for element in y:
#     if element > 1.1: element = 1
#     if element < 0.9: element = 1
# y = signal.savgol_filter(y, 61, 0)
# y = normalize(y)
show_plot(y)

corr = signal.correlate(x, y, mode = 'full') / 128

plt.plot(corr)
plt.show()

# sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
# sig_noise = sig + np.random.randn(len(sig))
# corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128
# clock = np.arange(64, len(sig), 128)
# fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
# ax_orig.plot(sig)
# ax_orig.plot(clock, sig[clock], 'ro')
# ax_orig.set_title('Original signal')
# ax_noise.plot(sig_noise)
# ax_noise.set_title('Signal with noise')
# ax_corr.plot(corr)
# ax_corr.plot(clock, corr[clock], 'ro')
# ax_corr.axhline(0.5, ls=':')
# ax_corr.set_title('Cross-correlated with rectangular pulse')
# ax_orig.margins(0, 0.1)
# fig.tight_layout()
# plt.show()