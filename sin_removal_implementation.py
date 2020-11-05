import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.fftpack import fft, ifft

t = np.linspace(0,1,1000, False)
sin_10hz = np.sin(2*np.pi*10*t)
sin_15hz = np.sin(2*np.pi*15*t)
noise = np.random.random_sample((len(t),))

clean_signal = sin_10hz + sin_15hz
signal = sin_10hz + sin_15hz + noise

plt.plot(noise)
plt.show()

plt.plot(clean_signal)
plt.show()

plt.plot(signal)
plt.show()