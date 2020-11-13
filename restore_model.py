# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler

def RestoreModel(model_name):
	new_model = tf.keras.models.load_model(modelname)
	return new_model

def over_limits_prediction(rmse,limits,test_size,y_test,test_predictions)
	for number in range(test_size)
		if rmse>limits:
			file_true='./spectrum_rmseover{0}/spectrum_rmse0.001over_{1}_true.csv'.format(limits,number+1)
				with open(file_true,mode='w') as f: 
					writer = csv.writer(f)
					writer.writerow(y_test[i])
		if rmse>limits:
			file_pred='./spectrum_rmseover{0}/spectrum_rmse0.001over_{1}_pred.csv'.format(limits,number+1)
				with open(file_pred,mode='w') as f:
					writer = csv.writer(f)
					writer.writerow(test_predictions[i])

def save_new_estimation_txt(B3F_Estimation,QQ_Estimation,New_Estimation)
	np.savetxt('/mnt/nfs_S65/Takayuki/B3F.txt',B3F_Estimation)
	np.savetxt('/mnt/nfs_S65/Takayuki/QQ.txt',QQ_Estimation)
	np.savetxt('/mnt/nfs_S65/Takayuki/New.txt',New_Estimation)
