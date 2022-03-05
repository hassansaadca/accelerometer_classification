import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.signal import welch


def show_single_sample():
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d', zlim = [-.7,.7])
    ax = plt.axes(projection='3d', zlim = [-.7,.7])#, figsize = (20,20))
    x = np.arange(1,20, .1)
    y = np.ones(x.shape[0])

    colors = [None,'b','r','k','b','k','r','b','k','r']

    rand_offset = np.random.randint(1,5, size = 10)
    for i in range(1,10):
        z = np.sin(x+rand_offset[i])
        ax.plot(x,y*i,z*.1, color = colors[i])
        
    return

def plot_premade_signals(x_values, amplitudes, frequencies, t_n = 10, N = 1000):
    T = t_n / N
    f_s = 1/T
    
    y_values = [amplitudes[i]*np.sin(2*np.pi*frequencies[i]*x_values) for i in range(0,len(amplitudes))]
    
    composite_y_value = np.sum(y_values, axis=0)
    
    colors = [None, 'orange','purple','green','red','brown','pink','blue','grey','cyan']
    fig, ax = plt.subplots(len(amplitudes)+1,1, figsize = (20,5), sharex = True, sharey = True)
    for i in range(len(y_values)):
        ax[i].plot(x_values, y_values[i], linewidth = 1.5, color = colors[i])
        ax[i].set_ylabel('Amp')
        if i == 0:
            ax[i].set_title('Signals of Various Frequencies and Amplitudes', size = 18)
    ax[-1].plot(x_values, composite_y_value, linewidth = 5, color = 'black')
    ax[-1].set_xlabel('Time (seconds)', size = 15)
    ax[-1].set_ylabel('Amp')
    plt.show()
    
    return (composite_y_value)



def show_true_signals(train_signals, train_labels):
    sample_signals=[]
    for i in range(1,7):
        sample_signals.append((train_signals[train_labels == i][0].T[3:6], i)) #gyroscope data

    with open('UCI_HAR/activity_labels.txt') as f:
        lines = f.readlines()

    activities = [i.split(' ')[1][:-1].replace('_', ' ') for i in lines]


    lab = ['x-direction','y-direction','z-direction']
    fig, ax = plt.subplots(6,1, figsize = (20, 10))
    for i in range(len(sample_signals)):
        for j in range(3):
            ax[i].plot(range(128), sample_signals[i][0][j], label = lab[j])
        ax[i].set_xlabel('Sample')
        ax[i].set_ylabel('Force')
    ax[0].legend()
    
    return
