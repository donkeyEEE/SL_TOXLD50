import pandas as pd
import models.utils as utils
from tabulate import tabulate


def fold_metric(get_data, data_path, metric):
    """
    使用get_data函数读取交叉折叠测试的结果，并且计算相应的指标
    Args:
        get_data (function): 用于读取数据的函数。
        data_path (str): 数据文件路径。
        metric (str): 需要计算的指标。
    
    Returns:
        pd.DataFrame: columns为模型名，index为metric*5
    """
    # 使用 get_data 方法获取数据并存储在 df 变量中
    df = get_data(data_path, 0)
    # 使用 utils.cal_df 函数计算 df 数据的度量统计值，并根据 metric 筛选出结果
    _df = utils.cal_df(df).loc[metric]
    # 重复 4 次以下过程，获取 data_path 中的数据并计算度量统计值
    for _i in range(4):
        # 计算当前迭代轮次的索引
        i = _i + 1
        # 从 data_path 中获取数据并存储在 df 中
        df = get_data(data_path, i)
        # 使用 utils.cal_df 函数计算 df 数据的度量统计值，并根据 metric 筛选出结果
        _df = pd.concat([_df, utils.cal_df(df).loc[metric]], axis=1)
    # 返回转置的汇总统计数据
    return _df.T

def fold_metric_average(get_data, data_path):
    """
    使用get_data函数读取交叉折叠测试的结果，计算相应的评估指标，最后平均得到五折平均测试结果
    Args:
        get_data (function): 用于读取数据的函数。
        data_path (str): 数据文件路径。
    Returns:
        pd.DataFrame: 包含不同模型的评估指标（MAE、RMSE、R2）的DataFrame
    """
    # 使用 get_data 方法获取数据并存储在 df 变量中
    df = get_data(data_path, 0)
    # 使用 utils.cal_df 函数计算 df 数据的度量统计值
    _df = utils.cal_df(df)
    # 重复 4 次以下过程，获取 data_path 中的数据并计算度量统计值的累加和
    for _i in range(4):
        # 计算当前迭代轮次的索引
        i = _i + 1
        # 从 data_path 中获取数据并存储在 df 中
        df = get_data(data_path, i)
        # 使用 utils.cal_df 函数计算 df 数据的度量统计值，并将结果累加到 _df 中
        _df += utils.cal_df(df)
    # 返回累加和的平均值，即数据的平均统计量
    return _df / 5

from tabulate import tabulate
def get_markdown(df):
    m = tabulate(df, headers='keys', tablefmt='pipe')
    for x in m.split('\n'):
        print(x)
        print('\n')