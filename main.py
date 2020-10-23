import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv
import DNN

#main
inputfile = './TrainingData/CTnumber_abs7500.csv'
outputfile = './TrainingData/spectrum7500_normalization.csv'

x_train,x_test,y_train,y_test=DNN.LoadData(inputfile,outputfile)

x_train_norm,x_test_norm=DNN.MinMax_normalization(x_train,x_test)

model = DNN.model_make()



#DNN.model_compile_Crossentropy(model)
DNN.model_compile_MSE(model)

stack=model.fit(x_train_norm,y_train,epochs=10,batch_size=32,validation_data=(x_test_norm,y_test))

DNN.compare_TV(stack)

test_predictions = model.predict(x_test_norm)

DNN.evaluation(test_predictions,y_test)

model.save('model.h5')
#np.set_printoptions(threshold=np.inf)
