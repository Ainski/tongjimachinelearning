import pandas as pd
import numpy as np


def create_theta(x_list):
    global theta
    theta=np.ones(len(x_list)+1)
    return theta

def clear_nan(train_df):
    train_df.dropna(inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    print(train_df.head())
    return train_df

def preprocess_data(x_train,y_train):
    global x_means ,x_stds
    global y_means,y_stds

    x_means= x_train.mean().tolist()#均值
    x_stds=x_train.std().tolist()#标准差
    
    y_means=y_train.mean().tolist()
    y_stds=y_train.std().tolist()
    
    normalized_x=(x_train-x_train.mean())/x_train.std()#归一化
    normalized_y=(y_train-y_train.mean())/y_train.std()

    return normalized_x,normalized_y,x_means,x_stds,y_means,y_stds

def linear_regression_train(X,y,theta=None,iter=0,max_iter=1000,lr=0.01):
    predictions=X.dot(theta)#预测值
    errors=predictions-y#误差
    gradient=X.T.dot(errors)/len(y)#梯度
    theta -= lr*gradient#更新参数
    iter+=1#迭代次数

    done = np.linalg.norm(gradient) < 1e-6 or iter >= max_iter#判断是否收敛
    return theta,done,iter

def ridge_regression_train(X, y,theta=None,iter=0,max_iter=1000,lr=0.01, alpha=0.5):

    theta,done,iter=linear_regression_train(X,y,theta,iter,max_iter,lr)
    theta[1:]-=alpha*lr*theta[1:]#正则化

    return theta,done,iter

def calculate_mse(x_test,y_test,theta,is_normalized=False):
    x_true=(x_test-x_means)/x_stds#归一化
    X = np.column_stack([np.ones(len(x_true)), x_true.values])
    y=y_test.values

    y_pred=X.dot(theta)#预测值
    
    if is_normalized and 'y_stds' in globals() and 'y_means' in globals():
        y_pred = y_pred * y_stds + y_means  # 反归一化
    
    y_true=y
    
    mse = np.mean((y_pred-y_true)**2)#计算均方误差
    print(y_pred)
    print(y_stds)
    print(y_means)
    print(y_true)
    return mse
regression_choices=["ridge_regression_train","linear_regression_train"]
regression_functions=[ridge_regression_train,linear_regression_train]