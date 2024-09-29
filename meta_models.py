# 修改2023813 准备将单个模型从集成模型类中分离出来，让框架更加稳定，并且可以支持多次运行单种模型
# 修改20240521，删除所有图相关的模型
import numpy as np
# import chemprop
import deepchem as dc
import deepchem.models
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
import pandas as pd
from sklearn import svm
import models.utils 
from deepchem.data import NumpyDataset
import warnings
import pickle
from typing import Dict
# import plot_parity, run_fun_SVR, run_fun_RF, run_fun_AFP_GAT,run_fun_DNN, dataloader_RF_SVR, dataloader_PytorchModel, ADFP_AC
# ===========
'''
meta_models 放入utils.model_sequential，组成模型基本结构
'''
# ===========

class meta_model:
    def __init__(self,
                id=0,
                save=False,
                model_save_files= "Stacking_model\\model_checkpoint",
                parament='',
                FP:str ='ECFP'
                ) -> None:
        self.id = id
        self.model_save_files = model_save_files
        self.mark = None # 用于最后显示模型结构的标识
        self.name = ""
        self.params = None
        self.model = None
        self.parament = parament
        self.FP=FP
    
    def get_model_mark(self):
        return f"id={self.id},{self.mark}"
    def run_model(self,train_set:NumpyDataset,test_set:NumpyDataset,trainmode:bool = False)->dict:
        # {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
        pass
    def get_params(self)->list:
        if self.FP !='ECFP':
            return self.FP
        else:
            return self.ECFP_Params
    def get_model_name(self)->str:
        return self.name+str(self.id)
    def predict(self, input:NumpyDataset)->Dict:
        if self.model is None:
            raise ValueError(f'{self.name} has not been trained')
        output = self.model.predict(input).reshape(-1)
        return {'prediction':output}
    def save(self,path:str='mp/model_0'):
        if os.path.exists(path) is not True:
            os.mkdir(path) 
        with open(f'{path}/data','wb') as file:
            pickle.dump(self,file)
    @staticmethod
    def load(path:str='mp/model_0'):
        try:
            with open(f'{path}/data','rb') as file:
                return pickle.load(file)
        except:
            return meta_DNN_TF.load(path)
    @staticmethod
    def Standard_datasets(train_set,test_set):
        from deepchem.data import NumpyDataset
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_s = scaler.fit_transform(train_set.X)
        X_s_test = scaler.transform(test_set.X)
        train_set = NumpyDataset(X=X_s,y=train_set.y)
        test_set = NumpyDataset(X=X_s_test,y=test_set.y)
        return train_set,test_set
from xgboost import XGBRegressor

class meta_XGB(meta_model):
    def __init__(self, 
                id=0, 
                save=False, 
                model_save_files="Stacking_model\model_checkpoint", 
                parament='',
                FP:str ='ECFP',
                ECFP_Params = [1024,2],
                **kwargs
                ) -> None:
        super().__init__(id, save, model_save_files, parament,FP=FP)
        self.name = f'meta_XGB'
        if 'ECFP_l' in kwargs.keys():
            del kwargs['ECFP_l']
            del kwargs['ECFP_r']
        
        self.ECFP_Params = ECFP_Params
        self.best_params = kwargs
        self.mark = f'XGB with '+self.FP
    
    def run_model(self,train_set:NumpyDataset,test_set:NumpyDataset,trainmode:bool = False)->dict:
        if self.FP == 'rdkit':
            train_set,test_set = self.Standard_datasets(train_set,test_set)
        # {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_test, 'test_pre': pre}
        model = XGBRegressor(**self.best_params)
        
        from deepchem.models import SklearnModel
        self.model = SklearnModel(model=model)
        self.model.fit(train_set)
        pre = self.model.predict(test_set)
        train_pre = self.model.predict(train_set)
        if trainmode:
            self.model= None
        return  {'train_true': train_set.y, 'train_pre': train_pre, 'test_true': test_set.y, 'test_pre': pre}



"""
<10.24>
修改为以tensorflow为主体的模型。因为使用deepchem和torch，拟合效果不好。虽然使用的是同一个结构。
可能出了些错误，暂时修改为tensor模型。
"""

import json
import os
import warnings

class meta_DNN_TF(meta_model):
    def __init__(self, 
                id=0, 
                save=False,
                ECFP_Params=[4096, 2],
                model_save_files="Stacking_model\\model_checkpoint",
                num_layers=3,
                layer_size_lis=None,
                learning_rate=0.001,
                batch_size=256,
                FP:str ='ECFP',) -> None:
        super().__init__(id, save, model_save_files,FP=FP)
        if FP == 'ECFP':
            self.mark = f"Deep neural network using ECFP={ECFP_Params},structure ={[ECFP_Params[0]] + [layer_size_lis for i in range(num_layers)]}"
        else:
            self.mark = f"Deep neural network using {FP},structure ={[ECFP_Params[0]] + [layer_size_lis for i in range(num_layers)]}"
        self.ECFP_Params = ECFP_Params
        self.name = "meta_DNN_TF"
        self.num_layers = num_layers
        self.layer_size_lis = layer_size_lis
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def run_model(self,train_set:NumpyDataset,test_set:NumpyDataset,trainmode:bool = False)->dict:
        from models.DNN_TF import build_DNN,remake_DNN
        if self.FP == 'rdkit' or self.FP == 'mordred':
            print(f'{self.mark} 对 {self.FP} 进行了标准化')
            train_set,test_set = self.Standard_datasets(train_set,test_set)
        
        _model = build_DNN(
            num_layers=self.num_layers,
            ECFP_Params=self.ECFP_Params,
            layer_size_lis=self.layer_size_lis,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size)
        recorder_DNN, DNN_model = remake_DNN(train_set, test_set, _model)       
        self.model = DNN_model 
        if trainmode:
            self.model= None
        return recorder_DNN

    def predict(self, input: NumpyDataset) -> Dict:
        if self.model is None:
            raise ValueError(f'{self.name} has not been trained')
        output = self.model.predict(input.X.astype(np.float32)).reshape(-1)
        return {'prediction': output}

    def save(self, path: str= 'mp/model_0'):
        if os.path.exists(path) is not True:
            os.mkdir(path) 
            
        # Save model
        self.model.save_model(path)
        
        # Save configuration to JSON
        config = {
            'id': self.id,
            'ECFP_Params': self.ECFP_Params,
            'name': self.name,
            'num_layers': self.num_layers,
            'layer_size_lis': self.layer_size_lis,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'mark': self.mark,
            'model_save_files': self.model_save_files,
            'FP':self.FP
        }
        config_path = os.path.join(path, 'metaconfig.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

    @staticmethod
    def load(path: str='mp/model_0'):
        from models.DNN_TF import DNN
        
        # Load configuration from JSON
        config_path = os.path.join(path, 'metaconfig.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize the class instance with loaded config
        instance = meta_DNN_TF(
            id=config['id'],
            ECFP_Params=config['ECFP_Params'],
            model_save_files=config['model_save_files'],
            num_layers=config['num_layers'],
            layer_size_lis=config['layer_size_lis'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            FP=config['FP']
        )
        instance.mark = config['mark']
        
        # Load the model
        instance.model = DNN.load_model(path)
        
        return instance

    
class meta_DNN(meta_model):
    def __init__(self, id=0, save=False,
            ECFP_Params=[4096, 2] ,
            model_save_files="Stacking_model\\model_checkpoint",
            num_layers=3) -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Deep neural network useing ECFP"
        self.ECFP_Params = ECFP_Params
        self.name = "meta_DNN"
        self.num_layers=num_layers
        
    def get_model_mark(self):
        return f"id={self.id},{self.mark} {self.ECFP_Params}"
    
    def get_DNN(self,train_set: dc.data.NumpyDataset, test_set: dc.data.NumpyDataset): # type: ignore
        # 定义模型
        layer_lis = []
        for i in range(self.num_layers):
            layer_lis.append(torch.nn.BatchNorm1d(num_features=self.ECFP_Params[0]))
            layer_lis.append(torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]))
            layer_lis.append(torch.nn.ReLU())
        
        layer_lis.append(torch.nn.Linear(int(self.ECFP_Params[0]), 1))
        
        DNN_model = torch.nn.Sequential(
            *layer_lis
        )
        optimizer = deepchem.models.optimizers.Adam()
        # from deepchem.models.losses 
        DNN_model = deepchem.models.TorchModel(DNN_model, 
                                            loss=dc.models.losses.L2Loss(),
                                            optimizer=optimizer,
                                            # batch_size=512,
                                            model_dir='{}/DNN_{}'.format(self.model_save_files,self.id)
                                            ) # type: ignore
        
        recorder_DNN, model_DNN = models.utils.run_fun_DNN(DNN_model, 
                                                        train_dataset=train_set, test_dataset=test_set,
                                                        ECFP_Params=self.ECFP_Params)
        if self.save:
            model_DNN.save_checkpoint(max_checkpoints_to_keep =1) 
        return recorder_DNN, model_DNN
    
    def load_predict(self,predict_data):
        print('DNN predict')
        DNN_model = torch.nn.Sequential(
                    torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.ECFP_Params[0], 1)
                )
        DNN_model = deepchem.models.TorchModel(DNN_model, loss=dc.models.losses.L2Loss(),model_dir='{}/DNN_{}'.format(self.model_save_files,self.id))# type: ignore
        DNN_model.restore()
        pre_DNN = DNN_model.predict(predict_data)
        return np.array(pre_DNN).reshape(-1)


    
class meta_RF(meta_model):
    def __init__(self, 
                id=0, 
                save=False,
                ECFP_Params=[4096, 2] ,
                n_estimators=300,
                min_samples_split=10,
                model_save_files="Stacking_model\\model_checkpoint",
                FP:str ='ECFP',) -> None:
        super().__init__(id, save, model_save_files,FP=FP)
        self.mark = f"Random Forest Regressor useing {FP},estimators = {n_estimators},min_samples_split ={min_samples_split}"
        self.ECFP_Params = ECFP_Params
        self.name = "meta_RF"
        self.n_estimators = n_estimators
        self.min_samples_split  = min_samples_split
    
    def run_model(self,train_set:NumpyDataset,test_set:NumpyDataset,trainmode:bool = False)->dict:
        # {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
        
        if self.FP == 'rdkit' or self.FP == 'mordred':
            train_set,test_set = self.Standard_datasets(train_set,test_set)
        model_RF = deepchem.models.SklearnModel(RandomForestRegressor(n_estimators=self.n_estimators, 
                                                                    min_samples_split=self.min_samples_split,
                                                                    n_jobs=3),
                        model_dir='{}/RF_{}'.format(self.model_save_files,self.id))
        
        recorder_RF,model = models.utils.run_fun_RF(model_RF, train_dataset=train_set, test_dataset=test_set,
                                        ECFP_Params=self.ECFP_Params)
        #if self.save:
        #    model_RF.save()
        self.model = model
        if trainmode:
            self.model= None
        return recorder_RF
    
    def load_predict(self,data):# type: ignore
        # print('RF predict')
        model_RF = deepchem.models.SklearnModel(RandomForestRegressor(n_estimators=181, min_samples_split=14),
                                                model_dir='{}/RF_{}'.format(self.model_save_files,self.id))
        print(self.id)
        model_RF.reload()
        pre_RF = model_RF.predict(data)        
        return np.array(pre_RF).reshape(-1) # type: ignore

class meta_SVR(meta_model):
    def __init__(self, 
                id=0, 
                save=False,
                ECFP_Params=[4096, 2],
                C=3,
                model_save_files="Stacking_model\\model_checkpoint",
                FP:str ='ECFP',) -> None:
        super().__init__(id, save, model_save_files,FP=FP)
        self.ECFP_Params = ECFP_Params
        self.name = "meta_SVR"
        self.C = C
        self.mark = f"Support vector regression useing {FP}, C={C}"

    def run_model(self,train_set:NumpyDataset,test_set:NumpyDataset,trainmode:bool = False)->dict:
        """
        {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
        """
        if self.FP == 'rdkit' or self.FP == 'mordred':
            train_set,test_set = self.Standard_datasets(train_set,test_set)
        model_SVR = deepchem.models.SklearnModel(svm.SVR(C=self.C), # type: ignore
                                                model_dir='{}/SVR_{}'.format(self.model_save_files,self.id)) # type: ignore
        recorder_SVR,model = models.utils.run_fun_RF(model_SVR, train_dataset=train_set, test_dataset=test_set,
                                                        ECFP_Params=self.ECFP_Params)
        self.model = model
        if trainmode:
            self.model= None
        return recorder_SVR
    def load_predict(self,predict_data):
        print('SVR predict')
        model_SVR = deepchem.models.SklearnModel(svm.SVR(C=1),
                                            model_dir='{}/SVR_{}'.format(self.model_save_files,self.id))
        model_SVR.reload()
        pre_SVR = model_SVR.predict(predict_data)
        return np.array(pre_SVR).reshape(-1)

import numpy as np
from typing import Dict
class L2_model():
    
    def __init__(self,id,save,model_save_files) -> None:
        self.id=id
        self.model=None
        self.name = None
    def get_model_mark(self):
        return f"id={self.id},{self.name}"
    
    def run_model_l2(self,recorder: Dict[(str,np.array)])->dict:
        # print('Start fitting RF in L2')
        try:
            self.model.fit(recorder['train_data'], recorder['train_true'])
            # pres = self.model.predict(recorder['test_data'])
        except Exception as e:
            raise ValueError(f"There has some problem when running {self.name}\n {e}")
        
        dic ={'y_train_pre':self.model.predict(recorder['train_data'])}
        return dic
    def predict(self,X:np.array) -> dict:
        dic ={'y_train_pre':self.model.predict(X)}
        return dic


class L2_RF(L2_model):
    def __init__(self, 
                id=0, 
                save=False, 
                model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = f"Random Forest Regressor for the second layer of Stacking_model"
        self.name = "L2_RF"
        self.model = RandomForestRegressor()

    
class L2_MLR(L2_model):
    def __init__(self, id=0, save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Multiple LinearRegression for the second layer of Stacking_model"
        self.name = "L2_MLR"
        self.model = LinearRegression()

class L2_SVR(L2_model):
    def __init__(self, id=0, save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Support vector regression for the second layer of Stacking_model"
        self.name = "L2_SVR"
        self.model = svm.SVR()

