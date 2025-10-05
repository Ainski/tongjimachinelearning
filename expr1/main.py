from datasets import load_dataset
from view import *
import pandas as pd
import numpy as np
from train import *
from download import *

dataset_path=data_path
def load_dataset_from_disk():
	train_df = pd.read_csv(f'{dataset_path}')
	return train_df
def createTrainAndTest(XRaw,YRaw,train_size=0.8,random_state=42):
	if random_state is not None:
		np.random.seed(random_state)
	
	n_samples = XRaw.shape[0]
	'''
	shape[0] 表示行数（即样本数量，对应代码中的n_samples）；
	shape[1] 表示列数（即特征数量）。
	'''
	
	n_train = int(train_size * n_samples)
	indices = np.random.permutation(n_samples)
	
	train_indices = indices[:n_train]
	test_indices = indices[n_train:]
	
	X_train = XRaw.iloc[train_indices]
	X_test = XRaw.iloc[test_indices]
	Y_train = YRaw.iloc[train_indices]
	Y_test = YRaw.iloc[test_indices]
	return X_train,X_test,Y_train,Y_test
	
	

if __name__ == '__main__':

	train_df = load_dataset_from_disk()
	clear_nan(train_df)
	y=train_df[y_label]
	x=train_df.drop(y_label,axis=1)
	
	x_train, x_test, y_train, y_test=createTrainAndTest(x,y)
	x_list=[col for col in x_train]



	app=QApplication(sys.argv)
	window=ScatterPlotWindow(x_train,y_train,x_list,x_test,y_test,y_label)
	window.show()
	
	sys.exit(app.exec_())
	