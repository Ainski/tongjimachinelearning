import kagglehub

import os
import shutil
from datasets import load_dataset

y_label="sales"
_target_path="./dataset"
data_path=_target_path+"/insurance.csv"
# https://www.kaggle.com/datasets/tohuangjia/advertising-simple-dataset?resource=download

# _dataset_path = kagglehub.dataset_download("mirichoi0218/insurance")
# if os.path.exists(_target_path):
# 	shutil.rmtree(_target_path)
#
# shutil.move(_dataset_path, _target_path)
# print(f"✅ 数据集已成功移动到: {_target_path}")
# def download_dataset():
# 	ds = load_dataset("mrseba/boston_house_price")
# 	ds.save_to_disk("./dataset")
#
# 	train_df = ds['train'].to_pandas()
#
# 	train_df.to_csv("./dataset/train.csv", index=False)
#
# 	print("Dataset downloaded and saved to disk.")