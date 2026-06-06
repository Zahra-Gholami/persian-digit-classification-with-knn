import numpy as np
import cv2
from scipy import io


def load_hoda(training_Sample_Size=1000 , test_sample_size=200, size=5):
    #load_dataset
    dataset=io.loadmat('./dataset/Data_hoda_full.mat')

    #test and training set
    x_train_original = np.squeeze(dataset['Data'][training_Sample_Size])
    y_train=np.squeeze(dataset['Label'][training_Sample_Size])

    x_test_original = np.squeeze(dataset['Data'][training_Sample_Size:training_Sample_Size+test_sample_size])
    y_test=np.squeeze(dataset['Label'][training_Sample_Size:training_Sample_Size+test_sample_size])

    #resize
    x_train_5by5=[cv2.resize(img,desize=(size,size)) for img in x_train_original]
    x_test_5by5=[cv2.resize(img, desize=(size,size)) for img in x_train_original]

    #reshape
    x_train=np.reshape(x_train_5by5,[-1,size**2])
    x_test=np.reshape(x_test_5by5,[-1,size**2])




    return x_train, y_train ,x_test, y_test
