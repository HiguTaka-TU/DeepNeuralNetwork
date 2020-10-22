import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv


def LoadData(inputfile,outputfile):
	f1=np.loadtxt(inputfile,delimiter=',')
	f2=np.loadtxt(outputfile,delimiter=' ')

	CT_values=np.array(f1)
	Spectrum=np.array(f2)


	x_train,x_test,y_train,y_test=train_test_split(CT_values,Spectrum,test_size=0.2,random_state=0)

	return x_train,x_test,y_train,y_test


def Zscore_nomalization(x_train,x_test,y_train,y_test):
	xmean=np.mean(x_train,axis=0)
	xstd=np.std(x_train,axis=0)

	x_train_norm=np.zeros((6000,14))
	x_test_norm=np.zeros((1500,14))

	for i in range(0,14):
		if xstd[i]!=0:
			x_train_norm[:,i]=(x_train[:,i]-xmean[i])/xstd[i]

	for i in range(0,14):
		if xstd[i]!=0:
			x_test_norm[:,i]=(x_test[:,i]-xmean[i])/xstd[i]
	
	return x_train_norm,x_test_norm

def MinMax_normalization(x_train,x_test,y_train,y_test):
	mmsc = MinMaxScaler()
	x_train_norm = mmsc.fit_transform(x_train)
	x_test_norm = mmsc.transform(x_test)
	np.set_printoptions(threshold=np.inf)

	return x_train_norm,x_test_norm


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

def model_compile_MSE():
	model.compile(optimizer='adam',
		loss='mean_squared_error')

def model_compile_Crossentropy():
	model.compile(optimizer='adam',
		loss='categorical_crossentropy')

def compare_TV():
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
	filename='./RMSE_max.png'
	plt.savefig(filename)
	plt.close()

	fig = plt.figure()
	plt.plot(test_predictions[RMSE_min])
	plt.plot(y_test[RMSE_min])
	filename='./RMSE_min.png'
	plt.savefig(filename)
	plt.close()


#main
inputfile = './TrainingData/CTnumber7500.csv'
outputfile = './TrainingData/spectrum7500_normalization.csv'

x_train,x_test,y_train,y_test=LoadData(inputfile,outputfile)

x_train_norm,x_test_norm=MinMax_normalization(x_train,x_test,y_train,y_test)

model = model_make()

model_compile_MSE()

stack=model.fit(x_train_norm,y_train,epochs=1000,batch_size=32,validation_data=(x_test_norm,y_test))

compare_TV()

test_predictions = model.predict(x_test_norm)

evaluation(test_predictions,y_test)

model.save('model.h5')
#np.set_printoptions(threshold=np.inf)
