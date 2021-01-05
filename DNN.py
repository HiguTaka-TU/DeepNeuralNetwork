import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import csv

#ファイルを読み込む
def load_data(inputfile,outputfile):
	CT_values=np.loadtxt(inputfile,delimiter=',')
	spectrum=np.loadtxt(outputfile,delimiter=' ')
	
	return CT_values,spectrum

def load_data_pd(input_file,output_file):
	CT_values = pd.read_csv(input_file,header=None) 
	spectrum = pd.read_csv(output_file,header=None)
	
	return CT_values,spectrum


#データの分割
def data_split(CT_values,spectrum,data2_fraction):
	x_1,x_2,y_1,y_2=train_test_split(CT_values,spectrum,test_size=data2_fraction,random_state=0)
	
	return x_1,x_2,y_1,y_2

#データフレームから列を削除
def drop_df(data_df,col):
	data=data_df.drop(col,axis=1)
	
	return data


#平均、標準偏差の計算
def calc_mean_std(x_train):
	xmean=np.mean(x_train,axis=0)
	xstd=np.std(x_train,axis=0)
	return xmean,xstd

#標準化
def standard_scaler(x_train,x_val,x_test):
	scaler = StandardScaler()

	scaler.fit(x_train)
	x_train_norm=scaler.fit_transform(x_train)
	x_val_norm   = scaler.transform(x_val)
	x_test_norm  = scaler.transform(x_test)
	
	np.set_printoptions(threshold=np.inf)
	
	return x_train_norm,x_val_norm,x_test_norm

#MinMaxScalerによる正規化
def minmax_normalization(x_train,x_val,x_test):
	scaler = MinMaxScaler()

	x_train_norm = scaler.fit_transform(x_train)
	x_val_norm   = scaler.transform(x_val)
	x_test_norm  = scaler.transform(x_test)
	np.set_printoptions(threshold=np.inf)

	return x_train_norm,x_val_norm,x_test_norm

#モデルの生成
def model_make():
	model = tf.keras.Sequential([
    	tf.keras.layers.Dense(13,input_shape=(13,)),
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

#モデルのコンパイル(損失関数はMSE)
def model_compile_mse(model):
	model.compile(optimizer='adam',
		loss='mean_squared_error')

#モデルのコンパイル(損失関数はクロスエントロピー)
def model_compile_crossentropy(model):
	model.compile(optimizer='adam',
		loss='categorical_crossentropy')

#アーリーストッピング
def early_stopping(patience,best):
	early_stpping = EarlyStopping(patience=patience,restore_best_weights=best)
	
	return early_stopping
	


#モデルをフィッティングする
def model_fit(model,x_train_norm,y_train,epochs,batch_size,x_val_norm,y_val):
	stack=model.fit(x_train_norm,y_train,
			epochs=epochs,
			batch_size=batch_size,
			validation_data=(x_val_norm,y_val))
			#callbacks=[early_stopping])	
	return stack

#学習曲線を表示
def compare_tv(stack,epochs,batch_size):
	loss = stack.history['loss']
	val_loss = stack.history['val_loss']
	loss_epochs = range(len(loss))
	
	fig=plt.figure()
	filename='./loss_epochs{0}_batch{1}.png'.format(epochs,batch_size)

	plt.plot(loss_epochs,loss,'blue',marker='^',label='training loss')
	plt.plot(loss_epochs,val_loss,'orange',marker='^',label= 'validation loss')
	plt.title('Training and Validation loss')
	plt.legend()

	plt.savefig(filename)

#入力から出力の推定を行う
def test_predictions(model,x_test_norm):
	predictions=model.predict(x_test_norm)
	return predictions

#推定結果と真値とのRMSEを計算
def evaluation_rmse(test_predictions,y_test):
	MSE=mean_squared_error(test_predictions,y_test)
	RMSE=np.sqrt(MSE)

	return RMSE
	
def rmse_minmax_number(RMSE):
	RMSE_max_number=np.argmax(RMSE)
	RMSE_min_number=np.argmin(RMSE)

	return RMSE_max_number,RMSE_min_number

def rmse_over_limit(RMSE,test_size,limit):
	number=[]
	for i in range(test_size):
		if RMSE[i]>limit:
			number.append(i)

	return number

#RMSEが最も大きい(最も外れた結果)を表示
def evaluation_RMSE_max_fig(RMSE_max_number):
	fig = plt.figure()
	plt.plot(test_predictions[RMSE_max_number])
	plt.plot(y_test[RMSE_max_number])
	filename_max='./RMSE_max.png'
	plt.savefig(filename_max)
	plt.close()

#RMSEが最も小さい(最も優れた結果)を表示
def evaluation_RMSE_min_fig(RMSE_min_number):
	fig = plt.figure()
	plt.plot(test_predictions[RMSE_min_number])
	plt.plot(y_test[RMSE_min])
	filename_min='./RMSE_min.png'
	plt.savefig(filename_min)
	plt.close()

#データのファイルへの書き込み
def write_csv(csv_name,data):
	with open(csv_name,mode='w') as f:
        	writer =csv.writer(f)
        	writer.writerow(data)
	
#モデルの保存
def model_save(model,model_name):
	model.save(model_name)

#メイン
def main():
	input_file ='./training_data/CTvalues/10000.csv'
	output_file ='./training_data/spectrum/10000.csv'
	
	#データの読み込み
	CT_values,spectrum=load_data_pd(input_file,output_file)
	pd.set_option('display.max_columns', 100)

	CT_values=drop_df(CT_values,5)

	#訓練データの分割
	x_train,x_2,y_train,y_2=data_split(CT_values,spectrum,data2_fraction=0.3)
	
	#検証、テストデータの分割
	x_val,x_test,y_val,y_test=data_split(x_2,y_2,data2_fraction=0.5)
	#真値の出力
	y_test.to_csv('true_value.csv',float_format='%.6f',header=False,index=False)
			
	#正規化を行う

	scaler='Zscore'
	#scaler='MinMax'
	if scaler=='Zscore':
		#標準化(Zスコア)
		x_train_norm,x_val_norm,x_test_norm=standard_scaler(x_train,x_val,x_test)
	if scaler=='MinMax':
		#正規化(MinMaxScaler)
		x_train_norm,x_val_norm,x_test_norm=minmax_normalization(x_train,x_val,x_test)
	
	#モデルの生成
	model=model_make()
	
	#モデルのコンパイル
	#loss='mse'
	loss='crossentropy'
	
	if loss=='mse':
		model_compile_mse(model)
	
	if loss=='crossentropy':
		model_compile_crossentropy(model) 
	
	#アーリーストッピング
	"""
	patience=10
	best='True'
	early_stopping=early_stopping(patience,best)
	"""
	
	#モデルフィット
	epochs,batch_size=1000,32
	stack=model_fit(model,x_train_norm,y_train,epochs,batch_size,x_val_norm,y_val)
	
	#学習曲線の表示
	compare_tv(stack,epochs,batch_size)

	#推定
	predictions=test_predictions(model,x_test_norm)
	#推定値の出力
	np.savetxt('predict_value.csv',predictions,fmt='%.6f',delimiter=',')

	#rmseの計算
	rmse_all=[]
	for i in range(1500):
		rmse=evaluation_rmse(predictions[i],y_test.iloc[i])
		rmse_all.append(rmse)
	np.savetxt('rmse_10000.csv',rmse_all,fmt='%.6f')
	
	#モデルの保存
	model_name='DNN_{0}_{1}.h5'.format(scaler,loss)
	model_save(model,model_name)
	
if __name__=="__main__":
	main()
