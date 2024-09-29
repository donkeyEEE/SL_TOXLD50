#
"""
超参数优化工具
输入：超参数选择，数据集，模型
输出：性能指标记录
"""
#

from typing import Dict,List,Union,Generator
from models.dataloader import DataLoader
import models.utils as utils
import json


class ModelHyperParams:
    """
    modelhyperparam = ModelHyperParam(model_name,papram_dic)
    dic_hyperparam:Dict = next(modelhyperparam.get_paramters())
    """
    def __init__(self,model_name:str,param_dic:Dict[(str,List[Union[float,int]])]) -> None:
        self.model_name = model_name
        self.param_dic = None
        self.init_paramter(param_dic)
    def get_paramters(self) -> Generator[Dict[str, Union[float, int]],None,None]:
        def cartesian_product(*lists):
            if not lists:
                yield []
                return
            for prefix in lists[0]:
                for suffix in cartesian_product(*lists[1:]):
                    yield [prefix] + suffix
        key_lis = list(self.param_dic.keys())
        lists = [self.param_dic[key] for key in key_lis]

        for product in cartesian_product(*lists):
            yield dict(zip(key_lis,product))
    
    def init_paramter(self,param_dic:dict):
        #self.checkparams(param_dic=param_dic) # 检测超参数是否正确
        # 使用param_dic初始化类参数
        self.param_dic = param_dic
        pass

    def checkparams(self,param_dic:dict):
        params_of_models={
            'RF':['ECFP_l','ECFP_r','n_estimators','min_samples_split'],
            'SVR':['ECFP_l','ECFP_r','C'],
            'DNN':['ECFP_l','ECFP_r','num_layers','learning_rate','batch_size']
        }
        try:
            parma_lis = params_of_models[self.model_name]
        except Exception as e:
            print(f'错误的模型名称，目前只支持:{params_of_models.keys()}')
        
        for _ in param_dic.keys():
            if _ not in parma_lis:
                raise ValueError(f'{_} is not support by {self.model_name}.The paramter of {self.model_name} is \n \t {parma_lis}')








# 性能记录器类，用于记录和计算模型的性能指标
class PerformanceRecorder:
    def __init__(self, param: dict):
        """
        初始化性能记录器。
        :param param: list, 模型训练的超参数列表
        """
        self.param_ = param  # 超参数
        self.param_str = str(list(param.values()))
        self.metric_train = {}  # 训练集的性能指标
        self.metric_test = {}  # 测试集的性能指标
        self.metric_average_train = {}  # 训练集的平均性能指标
        self.metric_average_test = {}  # 测试集的平均性能指标
        self.fold_count = 0  # 折数计数器

    def record_metrics(self, i, y_train, y_pred_train, y_test, y_pred_test):
        """
        记录当前折的训练和测试性能指标。
        :param i: int, 当前折数
        :param y_train: np.array, 训练集的真实值
        :param y_pred_train: np.array, 训练集的预测值
        :param y_test: np.array, 测试集的真实值
        :param y_pred_test: np.array, 测试集的预测值
        """
        # 计算并记录当前折的训练和测试性能指标
        self.metric_train[i] = self.calculate_metrics(y_train, y_pred_train)
        self.metric_test[i] = self.calculate_metrics(y_test, y_pred_test)
        self.fold_count += 1  # 增加折数计数

    def calculate_metrics(self, y_true, y_pred):
        """
        计算性能指标。
        :param y_true: np.array, 真实值
        :param y_pred: np.array, 预测值
        :return: dict, 包含MAE、RMSE和R2指标的字典
        """
        # 调用utils中的metric_r函数计算性能指标
        [MAE, RMSE, R2] = utils.metric_r(y_true, y_pred)
        return {"MAE": MAE, "RMSE": RMSE, "R2": R2}

    def calculate_average_metrics(self):
        """
        计算平均性能指标。
        """
        # 计算训练集和测试集的平均性能指标
        self.metric_average_train = self._calculate_average_metrics(self.metric_train, self.fold_count)
        self.metric_average_test = self._calculate_average_metrics(self.metric_test, self.fold_count)

    @staticmethod
    def _calculate_average_metrics(metrics, fold=5):
        """
        静态方法，用于计算字典中每个键的平均指标。
        :param metrics: dict, 包含每个折的性能指标的字典
        :param fold: int, 折数
        :return: dict, 包含平均性能指标的字典
        """
        # 使用字典推导式计算平均值
        average_metrics = {
            "MAE": sum(metrics[i]["MAE"] for i in range(fold)) / fold,
            "RMSE": sum(metrics[i]["RMSE"] for i in range(fold)) / fold,
            "R2": sum(metrics[i]["R2"] for i in range(fold)) / fold
        }
        return average_metrics

# 超参数记录器类，用于保存和导出超参数记录
class HyperparameterRecorder:
    def __init__(self,save_path=None,model_name:str=None):
        self.save_path = save_path
        self.records = HyperparameterRecorder.read_json(self.save_path)
        self.model_name = model_name
    def add_record(self, recorder:PerformanceRecorder):
        # 将PerformanceRecorder实例转换为字典
        recorder_dict = {
            'metric_average_train': recorder.metric_average_train,
            'metric_average_test': recorder.metric_average_test,
            'parameters': recorder.param_,
            'metric_train': recorder.metric_train,
            'metric_test': recorder.metric_test,
            'model':self.model_name,
        }
        self.records[recorder.param_str] = recorder_dict
    def save_to_json(self):
        """
        将超参数记录保存到JSON文件。
        :param filename: str, 保存文件的路径
        """
        # 将记录字典保存到JSON文件，格式化输出
        with open(self.save_path, 'w') as f:
            json.dump(self.records, f, indent=4)
    
    @staticmethod
    def read_json(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except:
            return {}
    @staticmethod
    def check_path(path:str):
        from pathlib import Path
        if bool(Path(path).suffix) == False:
            return path+'.json'
        
        if Path(path).suffix !='.json':
            raise KeyError(f'{path} is not a path to json')
        
        return path
    def plot_metrics(self, metric: str = 'R2', dataset_type: str = 'test'):
        import matplotlib.pyplot as plt
        """
        可视化不同超参数组合的性能指标。
        :param metric: str, 要展示的指标名称，默认为 'R2'。
        :param dataset_type: str, 数据集类型，'train' 或 'test'，默认为 'test'。
        """
        param_strs = []
        metric_values = []

        for record in self.records.values():
            param_strs.append(str(record['parameters'].values()))
            metric_values.append(record[f'metric_average_{dataset_type}'][metric])

        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.barh(param_strs, metric_values, color='skyblue')
        plt.xlabel(f'Average {metric}')
        plt.ylabel('Hyperparameters')
        plt.title(f'Hyperparameter Optimization Results ({dataset_type.capitalize()} Set)')
        plt.tight_layout()
        plt.show()
import copy
import pandas as pd
from deepchem.data import NumpyDataset
from models import meta_models
# 执行单个模型的优化任务
class ModelOptimizer:
    def __init__(self,
                dataloader:DataLoader,
                modelhyperparamters:ModelHyperParams,
                checkpoint_path:str='tmp.json',
                ) -> None:
        self.dataloader = dataloader
        self.checkpoint_path = checkpoint_path
        self.modelhyperparamters = modelhyperparamters

    
    def n_fold_test(self,model:meta_models)->HyperparameterRecorder:
        _path = HyperparameterRecorder.check_path(self.checkpoint_path)
        Hyper_Recorder = HyperparameterRecorder(_path)
        from tqdm import tqdm
        l = 0
        r =0 
        for dic_hyperparam in self.modelhyperparamters.get_paramters():
            print(f'Optimizing {self.modelhyperparamters.model_name} using hyperparamter:\n \t {dic_hyperparam}')
            if dic_hyperparam['ECFP_l'] != l and dic_hyperparam['ECFP_r'] != r:
                l = dic_hyperparam['ECFP_l'] 
                r= dic_hyperparam['ECFP_r'] 
                splitted_dataset = self.dataloader.get_fold_data(ECFP_Params=[dic_hyperparam['ECFP_l'],dic_hyperparam['ECFP_r']])
            
            recorder = PerformanceRecorder(dic_hyperparam)
            # 开始五折
            for i,(train_set,test_set) in tqdm(enumerate(splitted_dataset)):
                recorder = self.single_fold_test(train_set,test_set,
                                            model = model,
                                            paramters=dic_hyperparam,
                                            recorder=recorder,
                                            Hyper_Recorder=Hyper_Recorder,
                                            num_fold=i)
            if recorder.fold_count !=0:
                recorder.calculate_average_metrics()
                Hyper_Recorder.add_record(recorder=recorder)
                Hyper_Recorder.save_to_json()
        return Hyper_Recorder
    
    @staticmethod
    def single_fold_test(train_set:NumpyDataset,
                        test_set:NumpyDataset,
                        model :meta_models,
                        paramters:dict,
                        recorder:PerformanceRecorder,
                        Hyper_Recorder:HyperparameterRecorder,
                        num_fold:int = 1)->PerformanceRecorder:
        
        if recorder.param_str in Hyper_Recorder.records.keys():
            return recorder
        
        model = model(**paramters)
        df_recorder= model.run_model(train_set, test_set)
        recorder.record_metrics(
                num_fold,
                df_recorder['train_true'], df_recorder['train_pre'],
                df_recorder['test_true'], df_recorder['test_pre']
            )
        return recorder





