import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.signal import welch


def get_fourier_peaks(signal, samplingFrequency =50, npeaks = 5, demo = False):  
    
    samplingInterval = 1/samplingFrequency

    fhat = np.fft.fft(signal)/len(signal)
    fhat = 2*fhat[range(int(len(signal)/2))]

    num_samples    = len(signal)
    values      = np.arange(int(num_samples/2))

    timePeriod  = num_samples/samplingFrequency 
    frequencies = values/timePeriod
    
    #fill the rest of the array with zeros if we have fewer than 5 peaks
    res = frequencies[find_peaks(abs(fhat))[0][:npeaks]]
    ln = res.shape[0]
    padded_freq = np.append(res, np.zeros(npeaks-ln))
    
    res2 = abs(fhat)[find_peaks(abs(fhat))[0][:npeaks]]
    ln2 = res2.shape[0]
    padded_values = np.append(res2, np.zeros(npeaks-ln2))
    
    if demo:
        plt.plot(frequencies, abs(fhat))
        print(f'Frequencies: {padded_freq.round(2)}, \nCorresponding Amplitudes: {padded_values.round(2)}')
    else:
        return np.append(padded_freq, padded_values)

    
    
def get_autocorr_peaks(signal, T=1/50, N=128, npeaks = 5, demo = False):
    
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[len(result)//2:]
    
    autocorr_values = autocorr(signal)
    x_values = np.array([T * i for i in range(0, N)])
    
    res = x_values[find_peaks(autocorr_values)[0][:npeaks]]
    ln = res.shape[0]
    padded_time_delays = np.append(res, np.zeros(npeaks-ln))
    
    res2 = autocorr_values[find_peaks(autocorr_values)[0][:npeaks]]
    ln2 = res2.shape[0]
    padded_values = np.append(res2, np.zeros(npeaks-ln2))
    
    if demo:
        plt.plot(x_values, autocorr_values)
        print(f'Time Delays: {padded_time_delays.round(2)}, \nCorresponding Amplitudes: {padded_values.round(2)}')
    else:
        return np.append(padded_time_delays, padded_values)
