# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler

#モデルの保存
def RestoreModel(model_name):
	new_model = tf.keras.models.load_model(modelname)
	return new_model

#ある評価指標におけるある値を超えた(推定精度の悪い)結果をcsvファイルに書き込む
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

#推定結果をテキストファイルに保存
def save_new_estimation_txt(B3F_Estimation,QQ_Estimation,New_Estimation)
	file_name_B3F=''
	file_name_QQ=''
	file_name_New=''
	
	np.savetxt(file_name_B3F,B3F_Estimation)
	np.savetxt(file_name_QQ,QQ_Estimation)
	np.savetxt(file_name_New,New_Estimation)

#推定結果をcsvファイルに保存
def save_new_estimation_csv(predictions)
	save_name=''
	np.savetxt(save_name,predictions,fmt='%.6f')

if __name__=="__main__":
	
