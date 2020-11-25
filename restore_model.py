# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler

import DNN
#モデルの復元
def restore_model(model_name):
	new_model = tf.keras.models.load_model(model_name)
	return new_model

def load_data_pd(input_file):
	data = pd.read_csv(input_file,header=None)

	return data

#推定結果をテキストファイルに保存
def save_new_estimation_txt(B3F_Estimation,QQ_Estimation,New_Estimation):
	file_name_B3F='B3F_predict.csv'
	file_name_QQ='QQ_predict.csv'
	file_name_New='New_predict.csv'
	
	np.savetxt(file_name_B3F,B3F_Estimation,fmt='%.6f')
	np.savetxt(file_name_QQ,QQ_Estimation,fmt='%.6f')
	np.savetxt(file_name_New,New_Estimation,fmt='%.6f')

#推定結果をcsvファイルに保存
def save_new_estimation_csv(predictions):
	save_name=''
	np.savetxt(save_name,predictions,fmt='%.6f')

if __name__=="__main__":
	model=restore_model('./model/DNN_Zscore_crossentropy.h5')
	
	"""
	#pandasで読み込んで、waterの列を落とす
	interpolation=load_data_pd('../CTvalues/Interpolation.csv')
	interpolation=DNN.drop_df(interpolation,col=5)
	"""	

	#pd.set_option('display.max_columns', 100)
	
	input_file ='./training_data/CTvalues/10000.csv'
	output_file ='./training_data/spectrum/10000.csv'
	
	#データの読み込み
	CT_values,spectrum=DNN.load_data_pd(input_file,output_file)
	#pd.set_option('display.max_columns', 100)

	input_actual_file='actual_CTvalues_120kV.txt'
	actual_CTvalues = pd.read_csv(input_actual_file,header=None)
	
	actual_CTvalues=actual_CTvalues.transpose()
	
	
	#waterの列を落とす
	actual_CTvalues=DNN.drop_df(actual_CTvalues,5)
	CT_values=DNN.drop_df(CT_values,5)	

	#訓練データの分割
	x_train,x_2,y_train,y_2=DNN.data_split(CT_values,spectrum,data2_fraction=0.3)
	
	#検証、テストデータの分割
	x_val,x_test,y_val,y_test=DNN.data_split(x_2,y_2,data2_fraction=0.5)
	
	#標準化
	x_train_norm,x_val_norm,actual_CTvalues_norm=DNN.standard_scaler(x_train,x_val,actual_CTvalues)
	
	#推定
	
	predictions=DNN.test_predictions(model,actual_CTvalues_norm)
	#predictions=DNN.test_predictions(model,interpolation_norm)	
	predictions=DNN.test_predictions(model,actual_CTvalues_norm)	
	np.savetxt('120kV_predict.csv',predictions,fmt='%.6f')
	#save_new_estimation_txt(predictions[0],predictions[1],predictions[2])
