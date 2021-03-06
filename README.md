# Activity Recognition Using Smartphone Accelerometer Data

#### Background and Data:
In this project, I combine signal processing techniques common in mechanical engineering with machine learning tools to classify human movement with 91% accuracy. 

I accessed data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which consisted of accelerometer signals from Samsung Galaxy phones. 30 different individuals were recorded participating in different activities like walking, sitting, climbing up stairs, etc., and a total of 10,299 activities were recorded. Every activity consists of 9 different signals, and each signal is made up of 128 datapoints. The data therefore consists of a 3D matrix of 10,299 x 9 x 128 (number of samples x signals per sample x data points per signal).

Using signal processing techniques (specifically, the Fourier Transform and Autocorrelation), I transformed this 3D matrix into a 2D one which could be fed into a machine learning classifier.


#### Files:
`Samsung_Activity_Detection.ipynb` is the main notebook where data is read and a machine learning model is built. A brief demonstration of the Fourier Transform (the first signal processing technique we implement) is also provided.

`autocorr.ipynb`: A brief explanation of the implementation of autocorrelation, another signal processing technique which we implement to flatten our sample matrix.

`sigmod/plots.py`: a module which contains functions that help us visualize the 3D data and signal processing techniques

`sigmod/signal_processing.py`: a module in which I built the Fourier and Autocorrelation functions which help us flatten the 3-D data

`sigmod/load.py`: a module which contains a function to read in the data from the `UCI_HAR` folder


