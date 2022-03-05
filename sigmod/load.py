#credit to https://github.com/taspinar/siml for this loading code which I just put into a function
import numpy as np



def load_signal_data():

    def read_signals(filename):
        with open(filename, 'r') as fp:
            data = fp.read().splitlines()
            data = map(lambda x: x.rstrip().lstrip().split(), data)
            data = [list(map(float, line)) for line in data]
            data = np.array(data, dtype=np.float32)
        return data

    def read_labels(filename):        
        with open(filename, 'r') as fp:
            activities = fp.read().splitlines()
            activities = list(map(int, activities))
        return np.array(activities)

    INPUT_FOLDER_TRAIN = './UCI_HAR/train/InertialSignals/'
    INPUT_FOLDER_TEST = './UCI_HAR/test/InertialSignals/'

    INPUT_FILES_TRAIN = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt', 
                         'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                         'total_acc_x_train.txt', ""'total_acc_y_train.txt', 'total_acc_z_train.txt']

    INPUT_FILES_TEST = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt', 
                         'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                         'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

    train_signals, test_signals = [], []

    for input_file in INPUT_FILES_TRAIN:
        signal = read_signals(INPUT_FOLDER_TRAIN + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

    for input_file in INPUT_FILES_TEST:
        signal = read_signals(INPUT_FOLDER_TEST + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    LABELFILE_TRAIN = 'UCI_HAR/train/y_train.txt'
    LABELFILE_TEST = 'UCI_HAR/test/y_test.txt'
    train_labels = read_labels(LABELFILE_TRAIN)
    test_labels = read_labels(LABELFILE_TEST)

    return(train_signals, train_labels, test_signals, test_labels)