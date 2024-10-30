from typing import List,Dict,Union,Tuple
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from models.model_sequential import ModelSequential
from deepchem.data import NumpyDataset
from models.dataloader import DataLoader
import pickle

class DataManager_valid():
    pass

class DataManager_train():
    pass

class DataManager():
    def __init__(self,
                data:pd.DataFrame,
                model_squential:ModelSequential,
                seed = 5233,
                n_fold = 5) -> None:
        self.data_set:pd.DataFrame = None# 储存原始数据
        self.num_samples:int =None
        self.check_data(data)
        self.seed = seed
        self.model_squential = model_squential
        self.n_fold = n_fold
    def check_data(self,data:pd.DataFrame):
        self.data_set =data
        self.num_samples = data.shape[0]

    def trainmode(self)->DataManager_train:
        return DataManager_train(data=self.data_set,model_squential =self.model_squential,seed=self.seed,n_fold=self.n_fold
        )
    
    def validmode(self)->DataManager_valid:
        return DataManager_valid(data=self.data_set,
                model_squential = self.model_squential,
                seed = self.seed,)

class DataManager_train():
    def default_init(self):
        # Initialize default properties of the DataManager_train class.
        self.data_set: pd.DataFrame = None  # Store the original dataset.
        self.stack_pres: pd.DataFrame = None  # Store the predictions from cross-validation.
        self.n_fold_modelresult_Y: List[pd.DataFrame] = None  # Save results from fold testing.
        self.dic_dataloaders: Dict[str, List[Tuple[NumpyDataset, NumpyDataset]]] = {}  # Dictionaries for feature-wise datasets.
        self.n_fold: int = 5  # Number of folds for cross-validation in supervised learning.
        self.num_samples: int = None  # Number of samples in the dataset.
        self.df_stack_pres_and_labels: pd.DataFrame = None  # Training input and output for the second-layer model.

    def __init__(self, data: pd.DataFrame, model_squential: ModelSequential, seed=5233, n_fold=5) -> None:
        # Initialize the DataManager_train instance with a dataset, a sequential model, seed, and fold number.
        self.default_init()  # Call the default initialization.
        self.data_set = data
        self.num_samples = data.shape[0]
        self.seed = seed
        self.model_names: List[str] = []  # Get names of models in the sequential stack.
        self.n_fold = n_fold
        self.n_fold_modelresult_Y = [pd.DataFrame([], columns=self.model_names) for i in range(n_fold)]
        self.stack_pres = pd.DataFrame([], columns=self.model_names)
        self.dic_dataloaders_withoutsplit = {}
        
        
    def load_and_featurize(self, model_squential: ModelSequential):
        # Load and transform the dataset based on the parameters specified in the sequential model.
        param_lis = []
        for function in model_squential.functions:
            param = function.get_params()
            if param not in param_lis:
                param_lis.append(param)

        for _p in param_lis:
            from deepchem.splits import RandomSplitter
            split = RandomSplitter()
            if isinstance(_p,list):
                _dataset = DataLoader(data_set=self.data_set, ECFP_Params=_p).get_featurizer_data()
            else:
                _dataset = DataLoader(data_set=self.data_set,Descriptor_type=_p).get_featurizer_data()
            self.dic_dataloaders[str(_p)] = split.k_fold_split(_dataset, seed=self.seed, k=self.n_fold)
            self.dic_dataloaders_withoutsplit[str(_p)] = _dataset

    def get_dataset(self, param: str, num_fold: int = None) -> Tuple[NumpyDataset, NumpyDataset]:
        # Retrieve the dataset for a given parameter and fold number.
        if num_fold is None:
            return self.dic_dataloaders_withoutsplit[param]

        if param in self.dic_dataloaders:
            return self.dic_dataloaders[param][num_fold]
        else:
            raise ValueError(f'{param} is not in paramlist or it is not a string')

    def record_meta_model(self, model_id: str, num_fold: int, pres: Union[np.array, List], labels: Union[np.array, List] = None):
        # Record the predictions of a model for a specific fold, optionally with labels.
        self.n_fold_modelresult_Y[num_fold][model_id] = pres
        if labels is not None:
            self.n_fold_modelresult_Y[num_fold]['labels'] = labels

    def record_L2_model(self, pres: List, model_name: str) -> pd.DataFrame:
        # Record the predictions of the second-layer model.
        self.df_stack_pres_and_labels[f'pres_lables_{model_name}'] = pres
        return self.df_stack_pres_and_labels

    def get_stack_pres_and_labels(self) -> pd.DataFrame:
        # Combine predictions and labels from all folds to prepare for second-layer model training.
        _df = pd.DataFrame([])
        for _fold, df in enumerate(self.n_fold_modelresult_Y):
            _df = pd.concat([_df, df], axis=0)
        if self.df_stack_pres_and_labels is None:
            self.df_stack_pres_and_labels = _df
        return _df

    def eval(self):
        # Evaluate the model based on several metrics across all folds and record the performance.
        from models.utils import cal_df, metric_r
        class TrainRecorder():
            def __init__(self):
                self.models_performance_nfold_average: Dict[str, Dict[str, float]] = {}
                self.models_performance_nfold: List[Dict[str, Dict[str, float]]] = []
                self.L2models_performance: Dict[str, Dict[str, float]] = {}

            def record_nfold(self, n_fold_modelresult_Y: List[pd.DataFrame]):
                for _df in n_fold_modelresult_Y:
                    metric_df = cal_df(_df, label_name='labels')
                    self.models_performance_nfold.append(metric_df.to_dict())

            def cal_average(self):
                for _i, _performance_nfold in enumerate(self.models_performance_nfold):
                    if _i == 0:
                        df = pd.DataFrame(_performance_nfold)
                    else:
                        df += pd.DataFrame(_performance_nfold)
                df = df / 5
                self.models_performance_nfold_average = df.to_dict()

        recorder = TrainRecorder()
        recorder.record_nfold(self.n_fold_modelresult_Y)
        recorder.cal_average()

        for col_name in self.df_stack_pres_and_labels.columns:
            if 'L2' not in str(col_name):
                continue
            metric_array = metric_r(y_true=self.df_stack_pres_and_labels['labels'], y_pred=self.df_stack_pres_and_labels[col_name])
            metric_name = ['MAE', 'RMSE', 'R2']
            recorder.L2models_performance[col_name] = dict(zip(metric_name, metric_array))
        return recorder
    
    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path)->DataManager_train:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
        
    
class DataManager_valid():
    def default_init(self):
        self.data_set:pd.DataFrame = None# 储存原始数据
        self.testmode_modelresult_Y:pd.DataFrame = pd.DataFrame([]) # 测试的结果保
        self.withoutsplit_dic_dataloader:Dict[(str,NumpyDataset)]  ={}# 字典，特征化后的数据集,没有分割
        self.num_samples:int =None
        self.df_labels_testmode:pd.DataFrame =pd.DataFrame([])  # 第二层模型的测试集输入和输出 
        self.df_labels_trainmodel:pd.DataFrame = pd.DataFrame([])  # 第一层模型的训练输出 
        #  _df = {model.id for model in ModelSequential } +{'lables'} +{'pre_lables'}
    def __init__(self,
                data:pd.DataFrame,
                model_squential:ModelSequential,
                seed = 5233,
                n_fold = 5) -> None:
        self.default_init()
        self.seed =seed
        self.model_names:List[str] = []
        self.data_set =data
        self.num_samples = data.shape[0]

    def load_and_featurize(self,model_squential:ModelSequential):
        param_lis = []
        for _ in model_squential.functions:
            param = _.get_params()
            if param not in param_lis:
                param_lis.append(param)
        
        for _p in param_lis:
            if isinstance(_p,list):
                _dataset = DataLoader(data_set=self.data_set, ECFP_Params=_p).get_featurizer_data()
            else:
                _dataset = DataLoader(data_set=self.data_set,Descriptor_type=_p).get_featurizer_data()
            self.withoutsplit_dic_dataloader[str(_p)] = _dataset
    def get_dataset(self,param:str)->NumpyDataset:
        return self.withoutsplit_dic_dataloader[param]

    def record_meta_model(self,
                                model_id:str, # 模型编号
                                pres:Union[np.array,List],
                                labels:Union[np.array,List] =None,
                                ):
        # {model.id for model in ModelSequential } +{'lables'}
        self.testmode_modelresult_Y[model_id] = pres
        if labels is not None:
            self.testmode_modelresult_Y['labels'] = labels
        # raise warnings('增加异常处理，解决长度不一致的问题')
    
    def record_meta_model_trainmode(self,
                                model_id:str, # 模型编号
                                pres:Union[np.array,List],
                                labels:Union[np.array,List] =None,
                                ):
        # {model.id for model in ModelSequential } +{'lables'}
        self.df_labels_trainmodel[model_id] = pres
        if labels is not None:
            self.df_labels_trainmodel['labels'] = labels
        # raise warnings('增加异常处理，解决长度不一致的问题')
    
    def record_L2_model(self,pres:List,model_name:str):
        if self.df_labels_testmode is None:
            self.df_labels_testmode = self.testmode_modelresult_Y.copy()
        self.df_labels_testmode[model_name] = pres
        pass
    def get_pres_and_labels_testmode(self)->pd.DataFrame:
        return self.testmode_modelresult_Y
    def eval(self):
        # 在验证集上的指标
        # 1. 完整训练集上的训练结果(未实现)
        # 2. 使用元模型在验证集的效果
        # 3. 第二层模型在验证集上的效果
        from models.utils import cal_df,metric_r
        class TestRecorder():
            def __init__(self) -> None:
                # self.models_performance_ALL_trainset:Dict[(str,Dict[(str,float)])] = {}
                self.models_performance_validset:Dict[(str,Dict[(str,float)])] = {}
                self.L2models_performance_validset:Dict[(str,Dict[(str,float)])] = {}
            def record(self,df_labels_testmode:pd.DataFrame):
                metric_df = cal_df(df_labels_testmode,label_name='labels')
                dic = metric_df.to_dict()
                for _ in dic.keys():
                    if 'L2' in str(_):
                        self.L2models_performance_validset[_] = dic[_]
                    else:
                        self.models_performance_validset[_] = dic[_]
        TR = TestRecorder()
        TR.record(self.df_labels_testmode)
        return TR
    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path)->DataManager_valid:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
        