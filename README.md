# Activity Recognition Using Smartphone Accelerometer Data

#### Background and Data:
In this project, I accessed data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which consisted of accelerometer signals from Samsung phones. 30 different individuals were recorded participating in different activities like walking, sitting, climbing up stairs, etc, and a total of 10,299 activities were recorded. Every activity output 9 different signals, and each signal is made up of 128 datapoints. The data therefore consists of a 3D matrix of 10,299 x 9 x 128. 

Using signal processing techniques (specifically, the Fourier Transform and Autocorrelation), I transformed this 3D matrix into a 2D one which could be fed into a machine learning classifier. 

#### Files:
`UCI-HAR/`
|`test/`
||`IntertialSignals/`


----`subject_test.txt`


--`train/`


`Samsung_Activity_Detection.ipynb`: Main Notebook where data is read in and a machine learning model is trained. A brief overview of the Fourier Transform (the first signal processing technique) is also provided.

`autocorr.ipynb`: A brief explanation of the implementation of autocorrelation, another signal processing technique which we implement to flatten our sample matrix.

`plots`: a module which contains functions that help us visualize the 3D data and signal processing techniques

`load.py`: a module which contains a function to read in the data from the 




