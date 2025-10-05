import kagglehub

import os
import shutil
from datasets import load_dataset

'''
下载并加载波士顿房价数据集
'''
y_label="MEDV"
_target_path="./dataset"
data_path=_target_path+"/train.csv"
title="波士顿房价数据集"
def download_dataset_boston():
	ds = load_dataset("mrseba/boston_house_price")
	ds.save_to_disk("./dataset")

	train_df = ds['train'].to_pandas()

	train_df.to_csv("./dataset/train.csv", index=False)

	print("Dataset downloaded and saved to disk.")
if __name__ == '__main__':
	download_dataset_boston()

'''
下载并加载保险数据集
数据经过特殊订制，不可直接下载
存放在同级目录下的./insurance.csv当中，使用时请挪动到./dataset/insurance.csv
'''
# y_label = "charges"
# _target_path = "./dataset"
# data_path = _target_path + "/insurance.csv"
# title = "保险数据集"
#
#
# def download_dataset1():
# 	_dataset_path = kagglehub.dataset_download("mirichoi0218/insurance")
# 	if os.path.exists(_target_path):
# 		shutil.rmtree(_target_path)
#
# 	shutil.move(_dataset_path, _target_path)
# 	print(f"✅ 数据集已成功移动到: {_target_path}")
#
# if __name__ == '__main__':
# 	download_dataset1()

'''
下载并加载网店销售平台数据集
'''
# y_label = "sales"
# _target_path = "./dataset"
# data_path = _target_path + "/advertising.csv"
# title = "网点销售平台数据"
#
#
# def download_dataset2():
# 	_dataset_path = kagglehub.dataset_download("tohuangjia/advertising-simple-dataset")
# 	if os.path.exists(_target_path):
# 		shutil.rmtree(_target_path)
#
# 	shutil.move(_dataset_path, _target_path)
# 	print(f"✅ 数据集已成功移动到: {_target_path}")
#
#
# if __name__ == '__main__':
# 	download_dataset2()