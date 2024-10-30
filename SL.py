"""
SL.py

该模块实现了Super Learner集成学习框架，集成多种机器学习模型并通过交叉验证提高预测准确性。
在Evaluator.py中还包含提取特征重要性和评估模型性能的功能。

主要功能：
- 实现模型的堆叠以进行集成学习。
- 支持交叉验证和性能评估。
- 生成并保存预测结果。
- 可视化集成模型的表现。
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from models.meta_models import meta_model , L2_model
from models.model_sequential import ModelSequential
import deepchem
import pickle
from typing import List,Dict,Union,Tuple
from models.DataManager import DataManager,DataManager_train,DataManager_valid
from dataclasses import dataclass

@dataclass
class ModelsForPrediction():
    def __init__(self,model_lis:List[str]):
        self.model_choice = {m:False for m in model_lis}
        self.model_choice.update({'SL':False})
    def get_models_chosen(self):
        return list(filter(lambda key: self.model_choice[key] is True, self.model_choice))
    def chose_models(self,model_been_chosen:List[str]):
        for _m in model_been_chosen:
            self.model_choice[_m] = True
    def clean_choice(self):
        self.model_choice = {m:False for m in self.model_choice.keys()}
    
class SuperLearner():
    pass

class SuperLearner():
    # 储存第二层的模型
    
    def __init__(self,
                train_set:pd.DataFrame,
                valid_set:pd.DataFrame,
                model_sequential:ModelSequential,
                L2_model_sequential :ModelSequential = None,
                num_fold:int = 5,
                seed = 5233,
                save_path:str = 'tmp') -> None:
        self.model_sequential = model_sequential
        self.model_sequential.show_models() # 打印模型结构
        self.train_DataManager:DataManager_train= DataManager(train_set,model_squential=model_sequential,seed=seed,n_fold=num_fold).trainmode()
        self.valid_DataManager:DataManager_valid = DataManager(valid_set,model_squential=model_sequential,seed=seed,n_fold=1).validmode()
        self.num_fold = num_fold
        self.L2_model_sequential = L2_model_sequential
        self.L2_model_sequential.show_models()
        self.Second_Layer_Models:List[L2_model] = []
        self.save_path = save_path

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model_sequential']
        
        return state
    def show_models(self):
        self.model_sequential.show_models()
    def train(self):
        self.n_fold()
        self.train_second_layer()
        pass

    def test(self):
        self.valid_DataManager.load_and_featurize(self.model_sequential)
        self.train_DataManager.load_and_featurize(self.model_sequential)
        self.train_meta_models_in_valimode()
        self.test_second_layer()
        pass
    
    def predict(self,Models:ModelsForPrediction,input_data:List[str])->Dict:
        #self.check_ModelsForPrediction(Models)
        #self.check_input(input_data)
        df= pd.DataFrame({'Smiles':input_data,'LogLD':[0 for i in range(len(input_data))]})
        output_dic = {}
        from models.dataloader import DataLoader
        for _m in Models.get_models_chosen():
            if _m == 'SL':
                L2_input = []
                for i,metamodel in enumerate(self.model_sequential.get_model_lis()):
                    if hasattr(metamodel,'ECFP_Params'):
                        dl = DataLoader(data_set=df,ECFP_Params=metamodel.ECFP_Params,Descriptor_type=metamodel.FP)
                    else:
                        dl = DataLoader(data_set=df,Descriptor_type=metamodel.FP)
                    _p = metamodel.predict(dl.get_featurizer_data())['prediction']
                    output_dic[i] = _p
                    L2_input.append(_p)
                for L2model in self.Second_Layer_Models:
                    output_dic[f'{L2model.name}'] = L2model.predict(np.array(L2_input).T)['y_train_pre']
                continue            
            _model = self.model_sequential.index_model(_m)
            if hasattr(_model,'ECFP_Params'):
                dl = DataLoader(data_set=df,ECFP_Params=_model.ECFP_Params,Descriptor_type=_model.FP)
            else:
                dl = DataLoader(data_set=df,Descriptor_type=_model.FP)
            prediction = _model.predict(dl.get_featurizer_data())
            prediction = prediction['prediction']
            output_dic[_m] = prediction
        
        return output_dic

    
    def eval(self):
        VR = self.valid_DataManager.eval()
        TR = self.train_DataManager.eval()
        return {'ValidRecord':VR,'TrainRecord':TR}
    
    def get_ModelsChoice(self):
        print(self.model_sequential.get_model_names())
        import warnings
        warnings.warn('Not finish yet Prediction parse')
        return ModelsForPrediction(self.model_sequential.get_model_names())
    
    def train_meta_models_in_valimode(self):
        for _model in self.model_sequential.get_model_lis():
            _param = str(_model.get_params())
            train_dataset = self.train_DataManager.get_dataset(param=_param)
            valid_dataset = self.valid_DataManager.get_dataset(param=_param)
            _df = _model.run_model(train_set=train_dataset,test_set=valid_dataset)
            self.valid_DataManager.record_meta_model(_model.id,pres=_df['test_pre'],labels=_df['test_true'])
            self.valid_DataManager.record_meta_model_trainmode(_model.id,pres=_df['train_pre'],labels=_df['train_true'])

    def test_second_layer(self):
        _df = self.valid_DataManager.get_pres_and_labels_testmode().copy() #{model.id for model in ModelSequential } +{'lables'}
        for model_L2 in self.Second_Layer_Models:
            # record_L2 = {'train_data':np.array(_df.iloc[:,:-1]),'train_true':np.array(_df.iloc[:,-1])}
            dic_r = model_L2.predict(X=np.array(_df.drop(columns=['labels'])))
            self.valid_DataManager.record_L2_model(pres=dic_r['y_train_pre'],model_name=model_L2.name)

    def n_fold(self):
        # 加载并且特征化数据
        from tqdm import tqdm
        self.train_DataManager.load_and_featurize(self.model_sequential)
        for n_fold in tqdm(range(self.num_fold)):
            for _model in self.model_sequential.get_model_lis():
                if isinstance(_model,meta_model) is not True:
                    raise ValueError(f'{_model} is not meta_models')
                _param = str(_model.get_params())
                train_dataset ,test_dataset = self.train_DataManager.get_dataset(_param,n_fold)
                _df = _model.run_model(train_set=train_dataset,test_set=test_dataset,trainmode=True)
                # _df = {'train_true': train_set.y, 'train_pre': y_hat_train, 'test_true': test_set.y, 'test_pre': y_hat}
                self.train_DataManager.record_meta_model(_model.id,
                                                            num_fold = n_fold,
                                                            pres = _df['test_pre'],
                                                            labels = _df['test_true'])
    def train_second_layer(self):
        _df:pd.DataFrame = self.train_DataManager.get_stack_pres_and_labels().copy()
        if self.L2_model_sequential is None:
            raise ValueError(f"请先定义L2的模型")
        for model_L2 in self.L2_model_sequential.get_model_lis():
            if isinstance(model_L2,L2_model) is not True:
                raise ValueError(f'{model_L2} is not L2_model')
            record_L2 = {'train_data':np.array(_df.drop(columns=['labels'])),'train_true':np.array(_df['labels'])}
            df_r = model_L2.run_model_l2(recorder=record_L2)
            self.train_DataManager.record_L2_model(pres=df_r['y_train_pre'],model_name=model_L2.name)
            self.Second_Layer_Models.append(model_L2)
    
    def save(self, _path:str='tmp'):
        import os
        if os.path.exists(_path) is not True:
            os.mkdir(_path)
            
        file_path = f'{_path}/SL'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file) # tmp/SL
        self.model_sequential.save(f'{_path}/metamodels')

    @staticmethod
    def load(_path:str='_All_Test\model_pretrained')->SuperLearner:
        file_path = f'{_path}/SL'  # '_All_Test\model_pretrained/SL'
        with open(file_path, 'rb') as file:
            sl =  pickle.load(file)
        sl.model_sequential = ModelSequential.load(f'{_path}/metamodels') 
        return sl
