import warnings

warnings.filterwarnings("ignore")
# from tqdm import tqdm
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import MACCSkeys
# from rdkit import DataStructs
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
# import deepchem as dc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, recall_score, \
    roc_auc_score
import os
# from sklearn.decomposition import PCA, NMF
import pandas as pd
import deepchem.data
import deepchem.feat
from urllib.request import urlopen
from scipy.stats import linregress

import json
def read_json_file(file_path):
    """在模型储存文件中查询并且读取模型结构json文件

    Args:
        file_path (_type_): _description_

    Returns:
    dict: _description_
    """
    # 在文件夹中检索一个JSON文件是否存在，如果存在则读取，否则返回None：
    if os.path.exists(file_path):  # 检查文件是否存在
        with open(file_path, "r",encoding='gbk') as json_file:
            data = json.load(json_file)  # 读取JSON数据
        return data
    else:
        return None
    

    
def write_json(file_path,dic):
    # model_save_files = os.getcwd()+"/Stacking_model/model_checkpoint"
    with open(file_path, "w") as json_file:
        json.dump(dic, json_file, indent=4)
    # return "done"




dic = {
    "model_name":{
        "ensemble_num":int ,
        "args":[],
    }
}

'''
像torch.sequential一样将自定义结构传入集成函数
class FunctionContainer:
    def __init__(self,*args):
        self.functions = list(args) # 传递给这个参数的所有值都会被打包成一个元组

    def get_model_lis(self):
        # 返回模型列表
        
    
    def show_models(self):
        # 返回模型结构
        
'''

def metric_r(y_true, y_pred)->list:
    """
    :param y_true:
    :param y_pred:
    :return: MAE,RMSE,R2
    """
    return [mean_absolute_error(y_true, y_pred),
            mean_squared_error(y_true, y_pred, squared=False),
            r2_score(y_true, y_pred)]

# 导入必要的库
import pandas as pd

def cal_df(df:pd.DataFrame,label_name='true')->pd.DataFrame:
    """计算回归模型评估指标的函数
    Args:
        df (DataFrame): 包含真实值和模型预测结果的DataFrame,每一列代表真值序列或者某个模型的预测结果
    Returns:
        DataFrame: 包含不同模型的评估指标（MAE、RMSE、R2）的DataFrame
    
    传入一个包含真值和预测结果的df:
    ,true,DNN,RF,SVR
    0,2.285557309,2.1897602,2.645290746005392,2.70399725629787
    1,3.257438567,3.226643,3.1561349264111236,3.1575102100632653
    """
    # 获取DataFrame的列名
    name_lis = df.columns

    _2d_dir= {}  # 创建一个空字典，用于存储不同模型的评估指标
    """
    # 遍历每一列（除了第一列，即真实值列），计算评估指标并存储在字典中
    for _n in range(len(name_lis)-1):
        # 调用 metric_r 函数计算评估指标
        points_lis = metric_r(df[label_name], df.iloc[:, _n+1])
        # 将评估指标存储在字典中，字典的键为模型的列名
        _2d_dir[df.columns[_n+1]] = points_lis
    """
    for _n in df.columns:
        if _n == label_name:
            continue
        # 调用 metric_r 函数计算评估指标
        points_lis = metric_r(df[label_name], df[_n])
        # 将评估指标存储在字典中，字典的键为模型的列名
        _2d_dir[_n] = points_lis

    # 创建包含评估指标的DataFrame，行为指标（MAE、RMSE、R2），列为模型的列名
    _metric_df = pd.DataFrame(_2d_dir, index=["MAE", "RMSE", "R2"])
    
    # 返回包含评估指标的DataFrame
    return _metric_df

        

def cla(x):  # EPA标签
    x = 10 ** x
    if x < 500:
        return 0
    elif x < 5000:
        return 1
    return None


def metric_c(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: accuracy_score, recall_score, roc_auc_score
    """
    # y_pred 为概率值
    return [accuracy_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            roc_auc_score(y_true, y_pred)]


def metric_arr(y_true, y_pred, mode= 'regression')->list:
    """
    :param y_true:
    :param y_pred: 若为分类，需为概率值，不然计算不了ROC_AUC
    :param mode: regression or classification 取决于使用哪种模型
    :return: 返回长度为三的列表,MAE,RMSE,R2 or accuracy_score, recall_score, roc_auc_score
    """
    if mode == 'classification':
        # y_pred 为概率值
        return metric_c(y_true, y_pred)
    elif mode == 'regression':
        return metric_r(y_true, y_pred)


def cheat(y_true, y_pred):
    lis1 = []
    lis2 = []
    for i in range(len(y_true)):
        if abs(y_true[i] - y_pred[i]) < 1:
            lis1.append(y_true[i])
            lis2.append(y_pred[i])
    y_true = np.array(lis1)
    y_pred = np.array(lis2)
    return y_true, y_pred

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import font_manager
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import linregress
from typing import Optional


def plot_parity_plus(y_true: np.ndarray, y_pred: np.ndarray, name: str=None,axmin = -1,axmax = 6,
                    savefig_path: Optional[str] = None) -> list:
    """
    Plot a comparison graph between predicted and true values, displaying the regression line, its equation, and statistical metrics directly on the plot without a legend.

    Parameters:
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    name : str
        Title of the plot.
    y_pred_unc : Optional[np.ndarray]
        Optional array of prediction uncertainties, if provided, will be represented in the plot.
    savefig_path : Optional[str]
        If provided, the plot will be saved to this path.
    Returns:
    list
        Contains the slope and intercept of the regression line.
    """
    plt.figure(figsize=(8, 6))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    nd = np.abs(y_true - y_pred) / (axmax - axmin)

    cmap = plt.cm.get_cmap('cool')
    norm = plt.Normalize(nd.min(), nd.max())
    colors = cmap(norm(nd))

    plt.scatter(y_true, y_pred, c=colors, cmap=cmap, norm=norm)

    plt.plot([axmin, axmax], [axmin, axmax], '--', linewidth=2, color='navy', alpha=0.7)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    ax = plt.gca()
    ax.set_aspect('equal')

    font_path = 'C:/Windows/Fonts/times.ttf'
    font_prop = font_manager.FontProperties(fname=font_path, size=15)

    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    x_regression = np.linspace(axmin, axmax, 100)
    y_regression = slope * x_regression + intercept

    plt.plot(x_regression, y_regression, 'r-', linewidth=2)

    anchored_text = AnchoredText(f"Regression Line: $y = {slope:.2f}x + {intercept:.2f}$\np < 0.001\nRMSE = {rmse:.2f}\n$R^2 = {r2:.2f}$", loc="upper left", prop=dict(size=14), frameon=False)
    anchored_text.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    anchored_text.patch.set_facecolor('#F0F0F0')
    ax.add_artist(anchored_text)

    plt.xlabel('Observed Values', fontproperties=font_prop)
    plt.ylabel('Predicted Values', fontproperties=font_prop)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    # plt.tight_layout()
    plt.grid(color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

    if savefig_path:
        plt.savefig(savefig_path, dpi=600, bbox_inches='tight')

    #plt.show()
    return [slope, intercept]





def plot_parity(y_true, y_pred, name, y_pred_unc=None, savefig_path=None):
    axmin = min(min(y_true), min(y_pred)) - 0.05 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.05 * (max(y_true) - min(y_true))
    # y_true, y_pred = cheat(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    # compute normalized distance
    nd = np.abs(y_true - y_pred) / (axmax - axmin)
    # create colormap that maps nd to a darker color
    cmap = plt.cm.get_cmap('cool')
    norm = plt.Normalize(nd.min(), nd.max())
    colors = cmap(norm(nd))

    # plot scatter plot with color mapping
    sc = plt.scatter(y_true, y_pred, c=colors, cmap=cmap, norm=norm)

    # add colorbar
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel('Normalized Distance', fontsize=14, weight='bold')

    plt.plot([axmin, axmax], [axmin, axmax], '--', linewidth=2, color='red', alpha=0.7)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    ax = plt.gca()
    ax.set_aspect('equal')

    # 设置 x、y轴标签字体和大小
    font_path = 'C:/Windows/Fonts/times.ttf'  # 修改为times new roman的字体路径
    font_prop = font_manager.FontProperties(fname=font_path, size=15)

    at = AnchoredText(f"$MAE =$ {mae:.2f}\n$RMSE =$ {rmse:.2f}\n$R^2 =$ {r2:.2f} ",
                    prop=dict(size=14, weight='bold'), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    at.patch.set_facecolor('#F0F0F0')
    ax.add_artist(at)

    plt.xlabel('Observed Log(LD50)', fontproperties=font_prop)
    plt.ylabel('Predicted Log(LD50) by {}'.format(name), fontproperties=font_prop)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    plt.tight_layout()
    plt.grid(color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    if savefig_path:
        plt.savefig(savefig_path, dpi=600, bbox_inches='tight')
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

def plot_parity_plus_single(ax, y_true: np.ndarray, y_pred: np.ndarray, axmin=-1, axmax=6,
                            t_size= 15,a_size=16,c:str = 'cool'):
    """
    Plot a comparison graph between predicted and true values on a single axis, displaying the regression line, its equation, and statistical metrics directly on the plot without a legend.

    Parameters:
    ax : matplotlib.axes.Axes
        The axis on which to draw the plot.
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    axmin : int or float
        Minimum value for the axes.
    axmax : int or float
        Maximum value for the axes.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    
    nd = np.abs(y_true - y_pred) / (axmax - axmin)

    cmap = plt.cm.get_cmap(c)
    norm = plt.Normalize(nd.min(), nd.max())
    colors = cmap(norm(nd))

    ax.scatter(y_true, y_pred, c=colors, cmap=cmap, norm=norm)

    ax.plot([axmin, axmax], [axmin, axmax], '--', linewidth=2, color='navy', alpha=0.7)
    ax.set_xlim((axmin, axmax))
    ax.set_ylim((axmin, axmax))
    ax.set_aspect('equal')

    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    x_regression = np.linspace(axmin, axmax, 100)
    y_regression = slope * x_regression + intercept

    ax.plot(x_regression, y_regression, 'r-', linewidth=2)
    
    anchored_text = AnchoredText(f"Regression Line: $y = {slope:.2f}x + {intercept:.2f}$\np < 0.001\nRMSE = {rmse:.2f}\n$R^2 = {r2:.2f}$", loc="upper left", prop=dict(size=a_size), frameon=False)
    anchored_text.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    anchored_text.patch.set_facecolor('#F0F0F0')
    ax.add_artist(anchored_text)
    
    # 设置自定义的X轴刻度和标签
    x_ticks = [-1,  1,  3,  5,6]  # 刻度位置
    x_labels = [f'$10^{{{i}}}$' for i in x_ticks]  # 刻度标签

    ax.set_xticks(ticks=x_ticks, labels=x_labels, fontsize=t_size)
    ax.set_yticks(ticks=x_ticks, labels=x_labels, fontsize=t_size)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_linewidth(2)    # 顶部边框
    ax.spines['right'].set_linewidth(2)  # 右侧边框
    ax.spines['bottom'].set_linewidth(2) # 底部边框
    ax.spines['left'].set_linewidth(2)   # 左侧边框

    # 设置边框颜色
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

def plot_multiple_parity_plus(y_true_list: list, 
                            y_pred_list: list, 
                            axmin=-1, 
                            axmax=6, 
                            name_lis:list = [],
                            savefig_path: Optional[str] = None,
                            c = 'cool'):
    """
    Plot a 4x4 grid of parity plots on a single figure.

    Parameters:
    y_true_list : list of np.ndarray
        List of arrays containing the true values for each model.
    y_pred_list : list of np.ndarray
        List of arrays containing the predicted values for each model.
    axmin : int or float
        Minimum value for the axes.
    axmax : int or float
        Maximum value for the axes.
    savefig_path : Optional[str]
        If provided, the plot will be saved to this path.
    """
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    axs = axs.flatten()  # Flatten the 4x4 grid into a 1D array for easier iteration

    for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
        plot_parity_plus_single(axs[i], y_true, y_pred, axmin=axmin, axmax=axmax,c=c)
        axs[i].set_title(f'{name_lis[i]}',fontdict={'fontsize': 20, 'weight': 'bold'})

    # Remove empty subplots if there are fewer than 16 models
    for j in range(i+1, 16):
        fig.delaxes(axs[j])

    plt.tight_layout()

    if savefig_path:
        plt.savefig(savefig_path, dpi=600, bbox_inches='tight')

    plt.show()

# 使用示例：
# y_true_list 和 y_pred_list 是长度为16的列表，每个元素对应一个模型的真实值和预测值
# plot_multiple_parity_plus(y_true_list, y_pred_list)










def dataloader_AFP_default(df):
    """
    数据加载器，输入df，并将其smiles转换格式，返回dc.dataset
    """
    
    featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    smiles_l = list(df.smiles)
    x = featurizer.featurize(smiles_l)
    y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=x, y=y_data) # type: ignore
    return dataset


def dataloader_PytorchModel(df, featurizer):

    smiles_l = list(df.smiles)
    x = featurizer.featurize(smiles_l)    
    y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=x, y=y_data)  # type: ignore
    return dataset

def dataloader_PytorchModel_629(df:pd.DataFrame, featurizer):
    """  629 去除不能图转化的化合物
    用于GAT和AFP的数据加载器，因为都是pytorch模型，故命名为dataloader_PytorchModel
    输入df，并将其smiles转换格式，返回dc.dataset
    :param df: 含有smiles和LogLD列的df
    :param featurizer : 和模型对应的转换器
    :return:返回NumpyDataset，用于dc类模型训练；以及空元素在df中的行索引
    """
    def check_array(arr):
    # 从一个array的元素中找到所有empty元素，并返回一个位置列表
        location_lis = []
        for i in range(len(arr)):
            if type(arr[i]) != type(arr[0]):
                location_lis.append(i)
        return location_lis
    
    # featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    # 空元素位置
    smiles_l = df.smiles
    
    x = featurizer.featurize(list(smiles_l))    
    y_data = df.LogLD
    location_lis = check_array(x)
    
    # 获取空元素在df中的行索引
    empty_rows = [df.index[i] for i in location_lis]

    x = np.delete(x,location_lis)
    y_data = np.delete(np.array(y_data) ,location_lis)
    dataset = deepchem.data.NumpyDataset(X=x, y=y_data) # type: ignore
    return dataset, empty_rows

def dataloader_RF_SVR_default(df):
    """

    数据加载器，读取指定位置的数据，并将其smiles转换为ECFP格式，返回dc.dataset

    """
    
    featurizer = deepchem.feat.CircularFingerprint(size=4096, radius=2)
    smiles_l = list(df.smiles)
    ECFP_l = featurizer.featurize(smiles_l)
    ECFP_l = np.vstack(ECFP_l)  # type: ignore # 转二维ndarray
    y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=ECFP_l, y=y_data) # type: ignore
    return dataset


def dataloader_RF_SVR(df, ECFP_Params):
    """
    数据加载器，读取指定位置的数据，并将其smiles转换为ECFP格式，返回dc.dataset
    504 添加ECFP超参数修改功能，在run_fuc中也有修改
    504 添加降维功能
    """
    featurizer = deepchem.feat.CircularFingerprint(size=ECFP_Params[0], radius=ECFP_Params[1])
    smiles_l = list(df.smiles)
    ECFP_l = featurizer.featurize(smiles_l)
    ECFP_l = np.vstack(ECFP_l)  # type: ignore # 转二维ndarray 
    ## ==== 添加PCA降维功能
    """
    pca = PCA(n_components=int(ECFP_Params[0]/2))
    pca.fit(ECFP_l)
    ECFP_l = pca.transform(ECFP_l)
    """
    ## ====
    ## ==== 添加NMF降维功能
    """
    nmf = NMF(n_components=int(ECFP_Params[0]/2))
    ECFP_l = nmf.fit_transform(ECFP_l)
    """
    ## ====
    if "LogLD" not in df.columns:
        y_data = np.ones(df.shape[0])
    else:
        y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=ECFP_l, y=y_data) # type: ignore
    return dataset


def print_score(name_lis, score_lis):
    return 
    for i in range(len(name_lis)):
        print(name_lis[i], ' is ', score_lis[i])


def run_fun_AFP_GAT(model, train_dataset, test_dataset, mode_class='AFP', mode='regression', epoch=5):
    """
    用于AFP,GAT和GCN的训练函数，传入模型、训练集和测试集，通过默认参数控制数据加载器类型。
    设置任务类型以控制模型指标
    save为True时保存模型
    除杂方面，返回空元素位置列表，在集成函数中进行删除
    Args:
        model (_type_): 将要运行的模型类：dc.models
        train_dataset (_type_): 训练集
        test_dataset (_type_): 测试集
        mode_class (str, optional): 模型类型，用于区分AFP和GAT的特征转换器. Defaults to 'AFP'.
        mode (str, optional): 训练模式,"regression"of "classification". Defaults to 'regression'.
        epoch (int, optional): 训练轮数. Defaults to 5.
    Returns:
        _type_: _description_
    """
    # ============================== # 
    # 训练和验证模型
    loss = model.fit(train_dataset, nb_epoch=epoch)
    y_train = train_dataset.y
    
    
    train_pre = model.predict(train_dataset)
    
    
    y_val = test_dataset.y
    pre = model.predict(test_dataset)
    name_lis=[]
    if mode == 'regression':
        name_lis = ['test_rmse', 'test_mae', 'test_R2']
    if mode == 'classification':
        name_lis = ['test_acc', 'test_recall', 'test_roc']
    score_lis = metric_arr(y_val, pre, mode)
    print_score(name_lis, score_lis)
    def lis_to_array(lis):
        lisss= [ x[0] for x in lis]
        return np.array(lisss)
    # 保存fold的结果
    fold_record = {'train_true': y_train, 'train_pre': lis_to_array(train_pre), 'test_true': y_val, 'test_pre': lis_to_array(pre)}
    return fold_record, model


def run_fun_RF(model_RF, train_dataset, test_dataset, mode='regression', ECFP_Params=[4096, 2]):
    # train_dataset = dataloader_RF_SVR(train_dataset, ECFP_Params)
    # test_dataset = dataloader_RF_SVR(test_dataset, ECFP_Params)
    if True:
        # 训练和验证模型
        model_RF.fit(train_dataset)
        y_train = train_dataset.y
        train_pre = model_RF.predict(train_dataset)

        y_val = test_dataset.y
        pre = model_RF.predict(test_dataset)

        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)

        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
        
    return fold_record,model_RF



def run_fun_DNN(model_DNN, train_dataset, test_dataset,
                ECFP_Params=None ,mode='regression',nb_epoch=40):
    # 训练和验证模型
    model_DNN.fit(train_dataset, nb_epoch=nb_epoch)
    y_train = train_dataset.y
    train_pre = model_DNN.predict(train_dataset)

    y_val = test_dataset.y
    pre = model_DNN.predict(test_dataset)
    def lis_to_array(lis):
        lisss= [ x[0] for x in lis]
        return np.array(lisss)
    #if mode == 'regression':
    #        name_lis = ['test_rmse', 'test_mae', 'test_R2']
    #if mode == 'classification':
    #        name_lis = ['test_acc', 'test_recall', 'test_roc']
    #score_lis = metric_arr(y_val, pre, mode)
    # print_score(name_lis, score_lis)
    # 保存fold的结果
    fold_record = {'train_true': y_train, 'train_pre': lis_to_array(train_pre), 'test_true': y_val, 'test_pre': lis_to_array(pre)}

    return fold_record, model_DNN


def run_fun_SVR(model_SVR, train_dataset, test_dataset, mode='regression', ECFP_Params=[4096, 2]):

    if True:
        train_dataset = dataloader_RF_SVR(train_dataset, ECFP_Params)
        test_dataset = dataloader_RF_SVR(test_dataset, ECFP_Params)

        # 训练和验证模型

        model_SVR.fit(train_dataset)
        y_train = train_dataset.y
        train_pre = model_SVR.predict(train_dataset)

        y_val = test_dataset.y
        pre = model_SVR.predict(test_dataset)

        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)

        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
    return fold_record, model_SVR
