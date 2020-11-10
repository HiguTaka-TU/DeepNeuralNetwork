# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv


#ファイルの読み込み
def load_data(inputfile,outputfile):
	f_input=np.loadtxt(inputfile,delimiter=',')
	f_output=np.loadtxt(outputfile,delimiter=' ')

	ct_values=np.array(f_input)
	spectrum=np.array(f_output)

	x_train,x_test,y_train,y_test=train_test_split(ct_values,spectrum,test_size=0.2,random_state=0)

	return x_train,x_test,y_train,y_test

#平均、標準偏差の計算
def calc_mean_std(x_train):
	xmean=np.mean(x_train,axis=0)
	xstd=np.std(x_train,axis=0)
	
	return xmean,xstd

#Zスコアによる正規化
def zscore_nomalization(x_train,xmean,xstd):
	
	x_train_norm=np.zeros((x_train.shape[0],x_train.shape[1]))

	for i in range(0,x_train.shape[1]):
		if xstd[i]!=0:
			x_train_norm[:,i]=(x_train[:,i]-xmean[i])/xstd[i]

	return x_train_norm

#MinMaxScalerによる正規化
def minmax_normalization(x_train,x_test):
	mmsc = MinMaxScaler()
	x_train_norm = mmsc.fit_transform(x_train)
	x_test_norm = mmsc.transform(x_test)
	np.set_printoptions(threshold=np.inf)

	return x_train_norm,x_test_norm

#モデルの生成
def model_make():
	model = tf.keras.Sequential([
    	tf.keras.layers.Dense(14,input_shape=(14,)),
    	tf.keras.layers.Dense(6000,activation='relu'),
    	tf.keras.layers.Dense(4000,activation='relu'),
    	tf.keras.layers.Dense(2000,activation='relu'),
    	tf.keras.layers.Dense(1500,activation='relu'),
    	tf.keras.layers.Dense(1000,activation='relu'),
    	tf.keras.layers.Dense(750,activation='relu'),
    	tf.keras.layers.Dense(500,activation='relu'),
    	tf.keras.layers.Dense(150,activation='softmax')
    	],name='Spectrum_Estimation_model')

	model.summary()
	
	return model


def model_compile_mse(model):
	model.compile(optimizer='adam',
		loss='mean_squared_error')

def model_compile_crossentropy(model):
	model.compile(optimizer='adam',
		loss='categorical_crossentropy')

def compare_tv(stack):
	loss = stack.history['loss']
	val_loss = stack.history['val_loss']
	epochs = range(len(loss))
	
	fig=plt.figure()
	filename='./epoch1000_loss_batch32.png'

	plt.plot(epochs,loss,'b',label='training loss')
	plt.plot(epochs,val_loss,'r',label= 'validation loss')
	plt.title('Training and Validation loss')
	plt.legend()

	plt.savefig(filename)

def evaluation(test_predictions,y_test):
	MSE=mean_squared_error(test_predictions,y_test)
	RMSE=np.sqrt(MSE)
	
	RMSE_max=np.argmax(RMSE)
	RMSE_min=np.argmin(RMSE)

	fig = plt.figure()
	plt.plot(test_predictions[RMSE_max])
	plt.plot(y_test[RMSE_max])
	filename_max='./RMSE_max.png'
	plt.savefig(filename_max)
	plt.close()

	fig = plt.figure()
	plt.plot(test_predictions[RMSE_min])
	plt.plot(y_test[RMSE_min])
	filename_min='./RMSE_min.png'
	plt.savefig(filename_min)
	plt.close()

def main():
	inputfile = './TrainingData/CTnumber7500.csv'
	outputfile = './TrainingData/spectrum7500_normalization.csv'

	x_train,x_test,y_train,y_test=LoadData(inputfile,outputfile)

	x_train_norm,x_test_norm=MinMax_normalization(x_train,x_test)

	model = model_make()

	model_compile_MSE()

	stack=model.fit(x_train_norm,y_train,epochs=1000,batch_size=32,validation_data=(x_test_norm,y_test))

	compare_TV()

	test_predictions = model.predict(x_test_norm)

	evaluation(test_predictions,y_test)

	model.save('model.h5')
	#np.set_printoptions(threshold=np.inf)

if __name__=="__main__":
	main()
