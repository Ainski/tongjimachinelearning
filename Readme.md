# 2025年同济大学计算结科学与技术专业机器学习课程作业
## 作业一：线性回归

本作业已实现可以支持任意数据集。
本作业已实现支持任意算法。
### 数据集下载
目前已有三个数据集存放。分别是房价，保险，网点销售平台。如要加载某一数据集，只需要把其他的数据集注释掉即可。其中第二个数据集经过特殊调整，因为string类型的数据无法参与到计算当中去。

如果想要加载其他数据集，请保证不存在string类型的统计量。
```python
y_label = "charges"# 要求指明哪一列是y标签
_target_path = "./dataset"#要求指明保存路径
data_path = _target_path + "/insurance.csv"#要求指明哪个文件
title = "保险数据集"#为这个图起一个名字吧


def download_dataset1():
	_dataset_path = kagglehub.dataset_download("mirichoi0218/insurance")
	if os.path.exists(_target_path):
		shutil.rmtree(_target_path)

	shutil.move(_dataset_path, _target_path)
	print(f"✅ 数据集已成功移动到: {_target_path}")#必须是csv文件才可以被接受

if __name__ == '__main__':
	download_dataset1()
```
### 添加新的算法
书写新的回归算法梯度下降函数，并存放到66-67行的列表当中
```python
regression_choices=["ridge_regression_train","linear_regression_train"]
regression_functions=[ridge_regression_train,linear_regression_train]
```
### 使用方法
如需下载数据集，请先运行download.py
```shell
python download.py
```
然后运行main.py
```shell
python main.py
```

在程序中，一定要进行数据预处理，否则会导致程序结果不可预测。