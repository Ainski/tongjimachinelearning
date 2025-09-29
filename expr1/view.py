import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication,QMainWindow,QVBoxLayout,QHBoxLayout,
                             QComboBox,QLabel,QWidget,QSizePolicy,QPushButton)

from PyQt5.QtCore import pyqtSignal,Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from train import *
from PyQt5 import QtCore
from download import title
class ScatterPlotWindow(QMainWindow):
    '''
    散点图窗口类，支持通过下拉菜单切换X轴数据列
    使用PyQt5的信号与槽机制实现交互
    '''
    
    # 定义信号：当选择的特征改变时发出信号，携带特征名称
    feature_changed = pyqtSignal(str)
    regre_changed=pyqtSignal(str)
    
    def __init__(self,x_data,y_data,feature_names,x_test,y_test,y_label,parent=None):
        '''
        初始化散点图窗口
        
        
        :param x_data: 特征数据 (DataFrame)
        :param y_data: 目标变量数据 (Series)
        :param feature_names: 可用的特征名称列表
        :param parent: 父窗口
        '''
        
        super().__init__(parent)
        
        self.x_data = x_data
        self.y_data = y_data
        self.x_test=x_test
        self.y_test=y_test
        self.y_true=y_test
        self.y_label=y_label

        self.feature_names = feature_names
        self.current_feature = feature_names[0] if feature_names else ""
        self.theta=create_theta(self.feature_names)
        self.show_regre_result=False
        self.init_ui()
        self.connect_signals()
        self.update_plot()
    # def add_preprocess_button(self):
    #     '''
    #     添加预处理按钮
    #     :return:
    #     '''
    #     self.preprocess_btn=QPushButton('执行数据预处理',self)
    #     self.preprocess_btn.clicked.connect(self.run_preprocess)
    #     self.control_layout.insertWidget(1,self.preprocess_btn)
    def run_preprocess(self):
        '''
        执行数据预处理
        :return:
        '''
        try:
            (processed_x,processed_y,self.x_means,
             self.x_stds,self.y_means,self.y_stds)=(
                preprocess_data(self.x_data,self.y_data))
            self.x_data=processed_x
            self.y_data=processed_y
            
            self.update_plot()
            self.info_label.setText("数据预处理完成")
        except Exception as e:
            self.info_label.setText(f"数据预处理失败：{str(e)}")
            print(f"数据预处理失败：{str(e)}")
            raise e


    def init_ui(self):
        '''
        初始化UI
        :return:
        '''
        self.regressionchoices=regression_choices
        self.regressionfunctions=regression_functions
        self.current_regre=self.regressionchoices[0]
        self.setWindowTitle(title)
        self.setGeometry(100,100,800,600)
        
        # 中央控件
        central_widget=QWidget()
        self.setCentralWidget(central_widget)
        main_layout=QVBoxLayout(central_widget)
        
        # 控制面板
        self.up_layout=QVBoxLayout()
        self.control_layout=QHBoxLayout()
        
        # 创建控制部件
        self.label=QLabel("选择X轴数据列:")
        
        self.combo_box=QComboBox()
        self.combo_box.addItems(self.feature_names)

        '''
        这个方法会将combo_box添加到control_layout布局中，
        并将其插入到label后面的位置，
        这样combo_box就会显示在label后面。
        '''
        self.regre_combo=QComboBox()
        self.regre_combo.addItems(self.regressionchoices)

        self.control_layout.addWidget(self.combo_box)
        '''
        add_preprocess_button()方法添加了一个预处理按钮，
        当用户点击该按钮时，会触发run_preprocess()方法执行数据预处理。
        '''
        self.preprocess_btn=QPushButton('执行数据预处理',self)
        self.preprocess_btn.clicked.connect(self.run_preprocess)
        self.control_layout.addWidget(self.preprocess_btn)

        '''
        这个方法会将按钮添加到control_layout布局中，
        并将其插入到combo_box后面的位置，
        这样按钮就会显示在combo_box和regre_combo之间。
        '''
        self.control_layout.addWidget(self.regre_combo)
        self.control_layout.addStretch()
        '''
        这个方法会将按钮添加到control_layout布局中，
        并将其插入到combo_box后面的位置，
        这样按钮就会显示在combo_box和regre_combo之间。
        '''
        self.rerges_execute_btn=QPushButton('执行回归',self)
        self.rerges_execute_btn.clicked.connect(self.run_regre)
        self.control_layout.addWidget(self.rerges_execute_btn)
        #create message label
        self.info_label=QLabel("")
        self.up_layout.addWidget(self.info_label)
        self.up_layout.addLayout(self.control_layout)
        
        main_layout.addLayout(self.up_layout)
        
        self.figure=Figure(figsize=(8,6),dpi=100)
        self.canvas=FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas)
        
    def connect_signals(self):
        '''
        连接信号
        :return:
        '''
        self.combo_box.currentTextChanged.connect(self.on_feature_changed)
        self.regre_combo.currentTextChanged.connect(self.on_regre_changed)
        self.feature_changed.connect(self.on_feature_update)
        self.regre_changed.connect(self.on_regre_update)
    def on_regre_changed(self,regre_name):
        print("选择的回归算法：",regre_name)
        self.regre_changed.emit(regre_name)
    def on_regre_update(self,regre_name):
        self.current_regre=regre_name
        
    def on_feature_changed(self,feature_name):
        print("选择的特征列：",feature_name)
        self.feature_changed.emit(feature_name)
        
    def on_feature_update(self,feature_name ):
        self.current_feature=feature_name
        self.update_plot()
        
    def update_plot(self):
        self.figure.clear()
        ax=self.figure.add_subplot(111)
        
        x=self.x_data[self.current_feature]
        y=self.y_data
        
        ax.scatter(x,y,alpha=0.6,c='blue',edgecolors='black',linewidth=0.5)

        '''
        绘制回归直线
        '''
        if self.show_regre_result and self.theta is not None:
            try:
                # 获取当前特征索引
                feature_idx=self.feature_names.index(self.current_feature)
                x_mean=self.x_data.mean()

                # 生成预测数据（其他特征取平均值）
                x_pred=np.linspace(x.min(),x.max(),100)
                
                # 正确构造方式（其他特征取平均值）
                X_pred = np.column_stack([
                    np.ones_like(x_pred),  # 截距项
                    x_mean.values.repeat(len(x_pred)).reshape(-1, len(self.feature_names))
                ])
                X_pred[:, feature_idx + 1] = x_pred  # 替换当前特征列为预测值

                y_pred=X_pred.dot(self.theta)
                ax.plot(x_pred, y_pred, color='red', linewidth=2, 
                    label=f'y = {self.theta[0]:.2f} + {self.theta[feature_idx+1]:.2f}x')
                ax.legend()
            except Exception as e:
                print(f"绘图错误: {str(e)}")
        ax.set_title(f'{self.current_feature} vs {self.y_label}',fontsize=14,fontweight='bold')
        ax.set_xlabel(self.current_feature,fontsize=12)
        ax.set_ylabel(f'{self.y_label}',fontsize=12)
        
        ax.grid(True,alpha=0.3)
        self.canvas.draw()
        
    def get_current_feature(self):
        return self.current_feature
    
    def set_feature(self,feature_name):
        if feature_name in self.feature_names:
            self.combo_box.setCurrentText(feature_name)
            self.current_feature=feature_name
    def run_regre(self):
        '''
        执行回归
        :return:
        '''
        self.theta=create_theta(self.feature_names)
        
        self.current_iter=0
        self.max_iters=1000
        self.done=False
        
        X = np.column_stack([
            np.ones(len(self.x_data)),  # 使用样本数量创建截距项
            self.x_data[self.feature_names].values  # 包含所有特征
        ])
        y=self.y_data.values
        
        self.timer=QtCore.QTimer()
        
        self.timer.timeout.connect(lambda:self.train_step(X,y))
        self.timer.start(100)
        
        self.info_label.setText(f"开始训练({self.current_regre}) 迭代:0")
    def train_step(self,X,y):
        self.show_regre_result=True
        if self.done:
            self.timer.stop()
            self.info_label.setText(f"模型均方误差为：{calculate_mse(self.x_test,self.y_true,self.theta,is_normalized=True)}")
            return
        try:
            # if self.current_regre == 'linear_regression_train':
            #     self.theta,self.done,self.current_iter=(
            #         linear_regression_train(X,y,self.theta,self.current_iter,self.max_iters))
            # else:
            #     self.theta,self.done,self.current_iter=(
            #         ridge_regression_train(X,y,self.theta,self.current_iter,self.max_iters))
            self.theta,self.done,self.current_iter=(
                self.regressionfunctions[regression_choices.index(self.current_regre)](X,y,self.theta,self.current_iter,self.max_iters))
            self.info_label.setText(
                f"{self.current_regre} 迭代:{self.current_iter} "
                f"收敛:{self.done}"
            )
            self.update_plot()
        except Exception as e:
            self.timer.stop()
            self.info_label.setText(f"训练失败: {str(e)}")
            print(f"训练错误: {str(e)}")