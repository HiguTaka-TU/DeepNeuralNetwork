import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler
"""


B3F_norm[5]=0
QQ_norm[5]=0
New_norm[5]=0

B3F_norm=B3F_norm.reshape([1,14])
QQ_norm=QQ_norm.reshape([1,14])
New_norm=New_norm.reshape([1,14])
#print(B3F_norm)
#print(QQ_norm)
#print(New_norm)

"""
def RestoreModel():
	modelname='model.h5'
	new_model = tf.keras.models.load_model("model.h5")
	
	#New_Estimation = new_model.predict(New_norm)
	return new_model

def Test_Prediction(new_model,test_norm):
	predictions = new_model.predict(test_norm)
	return predictions

"""
fig = plt.figure()
x=np.arange(1500)
plt.scatter(x,rmse,c='k',s=0.0001)
filename='./rmse.png'
plt.savefig(filename)
plt.close()
"""

"""
with open('rmse.csv',mode='w') as f:
        writer =csv.writer(f)
        writer.writerow(rmse)

for i in range(1,1500):
	if rmse[i]>0.001:
		#print(i)
		#print(y_test[i])
		filename='./spectrum_rmseover0.001/spectrum_rmse0.001over_%d_true.csv' % i
		with open(filename,mode='w') as f: 
			writer = csv.writer(f)
			writer.writerow(y_test[i])
		filename2='./spectrum_rmseover0.001/spectrum_rmse0.001over_%d_pred.csv' % i
		with open(filename2,mode='w') as f2:
			writer = csv.writer(f2)
			writer.writerow(test_predictions[i])
"""

"""
np.savetxt('/mnt/nfs_S65/Takayuki/B3F.txt',B3F_Estimation)
np.savetxt('/mnt/nfs_S65/Takayuki/QQ.txt',QQ_Estimation)
np.savetxt('/mnt/nfs_S65/Takayuki/New.txt',New_Estimation)
"""
