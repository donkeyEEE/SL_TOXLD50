import warnings
import json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import nnls
import os
import pathlib as Path
import models.meta_models as meta_models
import models.utils as utils
from models.DB import db
from models._Hyper_utils import PerformanceRecorder,HyperparameterRecorder
#############
'''
ensemble_models_new：是堆叠模型框架，项目中期时重构的代码，可以支持三层堆叠，以及所有meta_models

nnls_SL:是SL_TOX最终框架，使用超级学习器的搭建流程。

'''

#############

class ensemble_models_new:
    def __init__(self,model_save_files = "Stacking_model/model_checkpoint") -> None:
        # self.remove_chem = []
        self.ECFP_Params = [2048,2]
        self.model_save_files = model_save_files
        self.lis_coef =None
        self.lis_res = None


    def train_meta_models(self,train: pd.DataFrame,test:pd.DataFrame,
                        model_squential:utils.model_sequential,save=False,plot = False ):
        """训练不同的模型并生成它们的预测结果。
        先根据传入的模型，转化数据集
        Args:
            train (pd.DataFrame): 训练集
            test (pd.DataFrame): 测试集
            model_squential (_type_): 模型结构
            save (bool): 是否保存
            plot (bool): 是否绘图
        Returns:
            _type_: _description_
        """
        
        models = model_squential.get_model_lis()
        # 生成模型保存目录
        self.create_checkpoint()
        # 生成模型结构的json
        self.model_structure_dic = self.model_struture_save(models=model_squential)
        # 指定要保存的文件路径
        file_path = f"{self.model_save_files}/model_structure.json"
        # 将数据写入JSON文件
        with open(file_path, "w") as json_file:
            json.dump(self.model_structure_dic, json_file, indent=4)  # 使用indent参数添加缩进，使数据更易读
            
        # 模型数据加载
        data_set_dic = self.model_dataload(models_lis=models,train_data=train,test_data=test)
        
        
        # 汇总真实值和使用的模型的训练以及测试结果
        dic_train = {}
        dic_test = {}
        dic_train['true'] = np.array(data_set_dic['train_data'].LogLD).reshape(-1)
        dic_test['true'] = np.array(data_set_dic['test_data'].LogLD).reshape(-1)
        
        for id,model in enumerate(models):
            if save:
                model.save = True
            if plot:
                pass
            # 修改实例的储存位置
            if hasattr(model, 'model_save_files'):
                setattr(model, 'model_save_files',self.model_save_files)
            # 实例没有指定的属性
            else:
                print(f"Error: {model} object has no attribute \'model_save_files\'")  # 输出: Error: 'MyClass' object has no attribute 'other_value'
                

            # model.model_save_files = self.model_save_files
            
            model.id = id
            recorder_dic = self.run_models(train,test,model,data_set_dic)
            #recorder_df(dic): 训练和预测结果
            #{'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
            # dic_train[f'{model.name}_{id}'] = recorder_dic.get('train_pre', None)
            # dic_test[f'{model.name}_{id}'] = recorder_dic.get('test_pre', None)

            if isinstance(recorder_dic, dict):
                dic_train[f'{model.name}_{id}'] = recorder_dic.get('train_pre', None)
                dic_test[f'{model.name}_{id}'] = recorder_dic.get('test_pre', None)
            else:
                # 处理 recorder_dic 不是字典的情况
                print(recorder_dic)
                print(f"{model} has not returned the correct result")
        # self.dic_train = dic_train
        # self.dic_test = dic_test
        self.L1_train_df =pd.DataFrame(dic_train)
        self.L1_test_df =pd.DataFrame(dic_test)
        self.L1_train_df.to_csv('{}/L1_train_data.csv'.format(self.model_save_files))
        self.L1_test_df.to_csv('{}/L1_test_data.csv'.format(self.model_save_files))

    def load_pretrained_metamodels(self,predict_df:pd.DataFrame,model_save_files="Stacking_model/model_checkpoint"):
        """加载的模型并生成它们的预测结果。
        Args:
            predict_df (pd.DataFrame): _description_
            model_save_files (str, optional): _description_. 
        """
        self.predict_df = predict_df
        # 读取paraments，按照其中id来加载运行模型
        file_path = f"{self.model_save_files}/model_structure.json"
        model_structure = self.read_json_file(file_path)
        print(file_path)
        
        # 生成model_sequential
        if isinstance(model_structure,dict):
            model_sequential = self.get_model_sequential(model_structure)
            model_sequential.show_models()
        else:
            return 
        
        # 加载数据
        data_set_dic = self.model_dataload_pre(models_lis=model_sequential.get_model_lis(),train_data=predict_df)
        
        # 模型加载并且预测
        recorder = self.load_model(model_struture=model_sequential,data_set_dic=data_set_dic)
            
        recorder_df = pd.DataFrame(recorder)
        recorder_df.to_csv('{}/predict_df.csv'.format(self.model_save_files))
        self.L1_predict_df = recorder_df
        
    def L2_training(self,L2_model_sequential : utils.model_sequential,train_data:pd.DataFrame = None ,test_data:pd.DataFrame = None,save=False,plot=False): # type: ignore
        
        # 读取数据
        
        train_data = self.L1_train_df
        
        test_data = self.L1_test_df
        
        recorder = {"train_data":train_data[list(train_data.columns)[1:]],
                    "train_true":train_data['true'],
                    "test_data":test_data[list(test_data.columns)[1:]],
                    "test_true":test_data['true'],
        }
        # 汇总真实值和使用的模型的训练以及测试结果
        dic_train = {'true':np.array(recorder['train_true']).reshape(-1)}
        dic_test = {'true':np.array(recorder['test_true']).reshape(-1)}
        
        # 加载模型
        
        # 生成模型结构的json
        self.model_structure_dic_L2 = self.model_struture_save(models=L2_model_sequential)
        # 指定要保存的文件路径
        file_path = f"{self.model_save_files}/model_structure_L2.json"
        # 将数据写入JSON文件
        with open(file_path, "w") as json_file:
            json.dump(self.model_structure_dic_L2, json_file, indent=4)  # 使用indent参数添加缩进，使数据更易读
        
        # 运行模型
        for id,model in enumerate(L2_model_sequential.get_model_lis()):
            
            if plot:
                pass
            # 修改实例的储存位置
            if hasattr(model, 'model_save_files'):
                setattr(model, 'model_save_files',self.model_save_files)
            # 实例没有指定的属性
            else:
                print(f"Error: {model} object has no attribute \'model_save_files\'")  # 输出: Error: 'MyClass' object has no attribute 'other_value'
            model.id = id
            if hasattr(model,'get_model') :
                train_pre , test_pre,origin_model= model.get_model(recorder)
                # print( model.get_model(recorder))
                dic_train[f'{model.name}_{model.id}'] = train_pre
                dic_test[f"{model.name}_{model.id}"] = test_pre
                if save:
                    self.save_models(f'{model.name}_{model.id}',origin_model)
            else:
                print(f"model{id},{model} is not the currect model in L2")
        self.L2_train_df =pd.DataFrame(dic_train)
        self.L2_test_df =pd.DataFrame(dic_test)
        self.L2_train_df.to_csv('{}/L2_train_data.csv'.format(self.model_save_files))
        self.L2_test_df.to_csv('{}/L2_test_data.csv'.format(self.model_save_files))
    def load_pretrained_L2models(self,model_save_files="Stacking_model/model_checkpoint"):
        
        # 获取第一层的数据,划分特征
        # predict_df = self.L1_predict_df
        X = self.L1_predict_df[list(self.L1_predict_df.columns)[1:]]
        X.columns = pd.Index([ f'meta_{_n}' for _n in X.columns]) # 为DataFrame的列名添加前缀"meta_"
        # print(X)
        
        # 记录第二层的预测数据
        dic = {}
        
        # 读取paraments，按照其中id来加载运行模型
        file_path = f"{self.model_save_files}/model_structure_L2.json"
        model_structure = self.read_json_file(file_path)
        # print(model_structure)
        
        # 无法生成model_sequential，直接加载原模型
        if isinstance(model_structure,dict):
            for index,_n in enumerate(model_structure.keys()):
                _d = model_structure[_n]
                name =  _d['Model Categories']
                _model = self.load_models(f'{name}_{index}')
                pres = _model.predict(X)
                dic[f'{name}_{index}'] = pres
        else:
            print("There has some problem when load L2_models")
            
        self.L2_predict_df = pd.DataFrame(dic)
    
    
    def L3_training(self,save=False,plot=True):

        X = self.L2_train_df[list(self.L2_train_df.columns)[1:]]
        y = self.L2_train_df['true']

        Xt = self.L2_test_df[list(self.L2_test_df.columns)[1:]]
        yt = self.L2_test_df['true']
        
        # 拟合NNLS模型
        
        coefficients, residuals = nnls(X, y)
        
        self.lis_coef = coefficients
        self.lis_res = residuals
        # 打印系数和残差
        # print('L3,Coefficients:', coefficients)
        # print('L3,Residuals:', residuals)
        
        if save:
            # 将参数储存为json格式
            dic ={"coaf":list(coefficients),"res":int(residuals)}
            file_path = f"{self.model_save_files}/model_structure_L3.json"
            # model_save_files = os.getcwd()+"/Stacking_model/model_checkpoint"
            with open(file_path, "w") as json_file:
                json.dump(dic, json_file, indent=4)  # 使用indent参数添加缩进，使数据更易读
        
        # 测试集
        test_NNLS = 0
        for i in range(3):
            test_NNLS += coefficients[i] * np.array(Xt.iloc[0:, i])
        self.L3_test_df = pd.DataFrame({'true': np.array(yt).reshape(-1),
                                        'pre': test_NNLS})

        # 训练集
        train_NNLS = 0
        for i in range(3):
            train_NNLS += coefficients[i] * np.array(X.iloc[0:, i])
        self.L3_train_df = pd.DataFrame({'true': np.array(y).reshape(-1),
                                        'nnls': train_NNLS})
        self.L3_train_df.to_csv('{}/L3_train_data.csv'.format(self.model_save_files))
        self.L3_test_df.to_csv('{}/L3_test_data.csv'.format(self.model_save_files))
        
    def load_pretrained_L3model(self,predict_df=None):
        #  check
        if self.lis_coef is None or self.lis_res is None:
            file_path = f"{self.model_save_files}/model_structure_L3.json"
            params_dic = self.read_json_file(file_path)
            if params_dic == None:
                print("There has some problem in load L3")
                return None
            # dic ={"coaf":coefficients,"res":residuals}
            self.lis_coef = params_dic['coaf']
            self.lis_res = params_dic['res']
        # 计算输出结果
        out_arr = 0
        for i in range(len(self.lis_coef)):
            k = self.lis_coef[i]
            r = self.lis_res
            print(r)
            print(k)
            out_arr += k * (np.array(self.L2_predict_df.iloc[0:, i]).reshape(-1))
        # print(out_arr)
        self.L3_predict_array = out_arr
        if predict_df is not None:
            self.predict_df = pd.DataFrame({'smiles': predict_df.smiles, 'pre_LogLD': out_arr})
        print('use L3_predict_array to get output')
    
    def save_models(self,path, model):
        print('===============')
        print(self.model_save_files)
        file = open('{}/L2/{}.pickle'.format(self.model_save_files,path), 'wb')
        pickle.dump(model, file)
        file.close()
    # 加载模型函数
    def load_models(self,path):
        with open('{}/L2/{}.pickle'.format(self.model_save_files, path), 'rb') as file:
            # 从文件中加载数据处理器
            model = pickle.load(file)
            file.close()
        return model
    
    def get_model_sequential(self,dic:dict) -> utils.model_sequential:
        """从结构json文件读取的字典中，生成meta模型结构

        Args:
            dic (dict): 字典
            dic ={
                    'id':{
                        'Model Categories': str (meta_xxx),
                        'paraments':list | None
                    }
                }
        Returns:
            utils.model_sequential: _description_
        """
        lis = []
        
        for i, model_id in  enumerate(dic.keys()) :
            _d = dic[model_id]
            model_name = _d['Model Categories']
            _model = getattr(meta_models,model_name)
            model = _model()
            if "paraments" in _d.keys():
                setattr(model,'ECFP_Params',_d["paraments"])
                # print(_d["paraments"])
            
            setattr(model,'model_save_files',self.model_save_files)
            setattr(model,'id' ,i)
            lis.append(model)
            # print(model.name)
        return utils.model_sequential(*lis)
    
    def load_model(self,model_struture:utils.model_sequential, data_set_dic:dict) -> dict:
        
        recorder = {'true':data_set_dic['true_data'].LogLD}
        for model in model_struture.get_model_lis():
            if isinstance(model,meta_models.meta_AFP):  #使用isinstance来检查一个对象是否是某个类的实例：
                data_set = data_set_dic['AFP_train_data']
                pre_arr = model.load_predict(data=data_set)
                recorder[f'AFP_{model.id}'] = pre_arr
            if isinstance(model,meta_models.meta_MPNN):
                pre_arr = model.load_predict(data_set_dic['true_data'])
                recorder[f'MPNN_{model.id}'] = pre_arr    
            if isinstance(model,meta_models.meta_RF):
                data_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
                pre_arr = model.load_predict(data=data_set)
                recorder[f'RF_{model.id}'] = pre_arr
            if isinstance(model,meta_models.meta_DNN):
                data_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
                pre_arr = model.load_predict(data_set)
                recorder[f'DNN_{model.id}'] = pre_arr        
            if isinstance(model,meta_models.meta_SVR):
                data_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
                pre_arr = model.load_predict(data_set)
                recorder[f'SVR_{model.id}'] = pre_arr
        return recorder
    
    
    @staticmethod
    def model_struture_save(models:utils.model_sequential) -> dict :
        """
        函数model_struture_save用于生成模型结构的摘要信息，并将其存储在一个字典中。
        Args:
            models (utils.model_sequential): _description_
        Returns:
            dict: 
                dic ={
                    'id':{
                        'Model Categories': str (meta_xxx),
                        'paraments':list | None
                    }
                }
        """
        dic = {}
        models.show_models()
        for i,model_obj in enumerate(models.get_model_lis()):
            _d = {}
            if hasattr(model_obj,'name'):
                _d['Model Categories'] = getattr(model_obj,'name')
            else:
                print(f'{model_obj} has no "name" attribute')
            if isinstance(model_obj, meta_models.meta_DNN) or \
                isinstance(model_obj, meta_models.meta_RF) or \
                isinstance(model_obj, meta_models.meta_SVR)or \
                isinstance(model_obj, meta_models.meta_DNN_TF):
                
                _d['paraments'] = getattr(model_obj,'ECFP_Params')

            dic[f'model_{i}'] = _d
        return dic

    @staticmethod
    # 读取模型参数
    def read_json_file(file_path):
        """在模型储存文件中查询并且读取模型结构json文件

        Args:
            file_path (_type_): _description_

        Returns:
            dict: _description_
        """
        # 在文件夹中检索一个JSON文件是否存在，如果存在则读取，否则返回None：
        if os.path.exists(file_path):  # 检查文件是否存在
            with open(file_path, "r") as json_file:
                data = json.load(json_file)  # 读取JSON数据
            return data
        else:
            return None
    def create_checkpoint(self):
        """生成模型保存的文件,地址为，代码运行目录下的self.model_save_files
        """
        # path = os.getcwd()+ '/'+ self.model_save_files
        path = self.model_save_files
        # path_MPNN = os.getcwd()+ '/{}/MPNN'.format(self.model_save_files)
        path_MPNN = '{}/MPNN'.format(self.model_save_files)
        if os.path.exists(path) == False:
            os.mkdir(path)
        if os.path.exists(path + '/L2') ==False:
            os.mkdir(path + '/L2')
        if os.path.exists(path + '/L3') ==False:
            os.mkdir(path + '/L3')
        if os.path.exists(path_MPNN) ==False:
            os.mkdir(path_MPNN)


    def model_dataload(self,models_lis,train_data:pd.DataFrame ,test_data:pd.DataFrame):
        
        """多模型数据加载，根据选择的模型结果选择性加载数据，返回字典
        Args:
            models_lis (list): utils.model_squential.get_model_lis()
        """
        # has_person_instance = any(isinstance(obj, Person) for obj in object_list)
        output_dic={'train_data':train_data , 'test_data':test_data}
        if any(isinstance(obj, meta_models.meta_AFP) for obj in models_lis):
            train_set,train_empty_lis = meta_models.dataloader_AFP(train_data)
            test_set,test_empty_lis = meta_models.dataloader_AFP(test_data)
            # 删除无法转化的化合物
            train_data = train_data.drop(index=pd.Index(train_empty_lis))
            test_data = test_data.drop(index=pd.Index(test_empty_lis))
            output_dic['AFP_train_data'] = train_set
            output_dic['AFP_test_data'] = test_set
        if any(isinstance(obj, meta_models.meta_GAT) for obj in models_lis) or any(isinstance(obj, meta_models.meta_GCN) for obj in models_lis):
            train_set,train_empty_lis = meta_models.dataloader_GAT_GCN(train_data)
            test_set,test_empty_lis = meta_models.dataloader_GAT_GCN(test_data)
            output_dic['GAT_GCN_train_data'] = train_set
            output_dic['GAT_GCN_test_data'] = test_set
        if any(isinstance(obj, meta_models.meta_DNN) for obj in models_lis) or \
            any(isinstance(obj, meta_models.meta_RF) for obj in models_lis) or \
            any(isinstance(obj, meta_models.meta_SVR) for obj in models_lis)or \
            any(isinstance(obj, meta_models.meta_DNN_TF) for obj in models_lis):
            
            # 检索统计ECFP参数的种类
            train_set_dic={}
            test_set_dic={}
            for obj in models_lis:
                if hasattr(obj,'ECFP_Params'):
                    ECFP_Params = getattr(obj,'ECFP_Params')
                    train_set = meta_models.dataloader_DNN_RF_SVR(train_data,ECFP_Params)
                    test_set = meta_models.dataloader_DNN_RF_SVR(test_data,ECFP_Params)
                    train_set_dic[f'{ECFP_Params}'] = train_set
                    test_set_dic[f'{ECFP_Params}'] = test_set

            output_dic['RF_DNN_SVR_train_data'] = train_set_dic # type: ignore
            output_dic['RF_DNN_SVR_test_data'] = test_set_dic # type: ignore
        
        return output_dic

    def model_dataload_pre(self,models_lis,train_data:pd.DataFrame):
        
        """多模型数据加载，根据选择的模型结果选择性加载数据，返回字典
        Args:
            models_lis (list): utils.model_squential.get_model_lis()
        """
        # has_person_instance = any(isinstance(obj, Person) for obj in object_list)
        output_dic={}
        if any(isinstance(obj, meta_models.meta_AFP) for obj in models_lis):
            train_set,train_empty_lis = meta_models.dataloader_AFP(train_data)
            # 删除无法转化的化合物
            train_data = train_data.drop(index=pd.Index(train_empty_lis))
            output_dic['AFP_train_data'] = train_set

        if any(isinstance(obj, meta_models.meta_GAT) for obj in models_lis) or any(isinstance(obj, meta_models.meta_GCN) for obj in models_lis):
            train_set,train_empty_lis = meta_models.dataloader_GAT_GCN(train_data)

            output_dic['GAT_GCN_train_data'] = train_set

        if any(isinstance(obj, meta_models.meta_DNN) for obj in models_lis) or \
            any(isinstance(obj, meta_models.meta_RF) for obj in models_lis) or \
            any(isinstance(obj, meta_models.meta_SVR) for obj in models_lis):
            
            # 检索统计ECFP参数的种类
            train_set_dic={}

            for obj in models_lis:
                if hasattr(obj,'ECFP_Params'):
                    ECFP_Params = getattr(obj,'ECFP_Params')
                    train_set = meta_models.dataloader_DNN_RF_SVR(train_data,ECFP_Params)
                    train_set_dic[f'{ECFP_Params}'] = train_set

            output_dic['RF_DNN_SVR_train_data'] = train_set_dic # type: ignore
        
        output_dic['true_data']= train_data
        return output_dic

    def merge_df(self,recorder_lis):
        # 合并recorder_lis ,返回两个L1_train_df 和L1_test_df
        # recoder_lis中是recorder_dic(dic): 训练和预测结果{'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}

        # 初始化存储结果的列表
        L1_train_dfs = []
        L1_test_dfs = []

        for recorder_dic in recorder_lis:
            train_true = recorder_dic['train_true']
            train_pre = recorder_dic['train_pre']
            test_true = recorder_dic['test_true']
            test_pre = recorder_dic['test_pre']

            # 构造DataFrame
            train_df = pd.DataFrame({'True': train_true, 'Predicted': train_pre})
            test_df = pd.DataFrame({'True': test_true, 'Predicted': test_pre})

            L1_train_dfs.append(train_df)
            L1_test_dfs.append(test_df)

        # 合并所有的DataFrame
        L1_train_df = pd.concat(L1_train_dfs, axis=0)
        L1_test_df = pd.concat(L1_test_dfs, axis=0)

        return L1_train_df, L1_test_df           
                
            
    @staticmethod
    def delete_reuse(df, index_lis):
        # 删除异常化合物
        # df.reset_index(inplace=True)
        # df.drop(df.columns[0],axis = 1)
        # df.drop("level_0", axis=1, inplace=True)
        df = df.drop(index_lis)
        # df.reset_index(inplace=True)
        # df = df.drop(df.columns[0:2],axis = 1)
        
        # df.drop("level_0", axis=1, inplace=True)
        return df
    def run_models(self,train_data,test_data,model,data_set_dic:dict):
        """运行单个子模型
        Args:
            train(pd.DataFrame): 训练集
            test (pd.DataFrame): 测试集
            model (_type_): _description_
        Return:
            recorder_xxx(dic): 训练和预测结果
                {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
        """
        name = getattr(model,'name',None)
        
        if name == None:
            return dict([])
        elif isinstance(model,meta_models.meta_AFP):  #使用isinstance来检查一个对象是否是某个类的实例：
            train_set = data_set_dic['AFP_train_data']
            test_set = data_set_dic['AFP_test_data']
            print(type(train_set))
            recorder_AFP, model_AFP = model.get_AFP(train_set,test_set)
            print(recorder_AFP)
            return recorder_AFP
        elif isinstance(model,meta_models.meta_GAT):
            train_set = data_set_dic['GAT_GCN_train_data']
            test_set = data_set_dic['GAT_GCN_test_data']
            recorder_GAT, model_GAT = model.get_GAT(train_set,test_set)
            return recorder_GAT
        elif isinstance(model,meta_models.meta_GCN):
            train_set = data_set_dic['GAT_GCN_train_data']
            test_set = data_set_dic['GAT_GCN_test_data']
            recorder, model = model.get_GCN(train_set,test_set)
            return recorder
        elif isinstance(model,meta_models.meta_MPNN):
            recorder = model.get_MPNN(train_data,test_data)
            return recorder
        elif isinstance(model,meta_models.meta_RF):
            train_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
            test_set = data_set_dic['RF_DNN_SVR_test_data'][f'{model.ECFP_Params}']
            recorder , model = model.get_RF(train_set,test_set)
            return recorder
        elif isinstance(model,meta_models.meta_DNN):
            train_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
            test_set = data_set_dic['RF_DNN_SVR_test_data'][f'{model.ECFP_Params}']
            recorder , model =model.get_DNN(train_set,test_set)
            return recorder
        
        elif isinstance(model,meta_models.meta_DNN_TF):
            train_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
            test_set = data_set_dic['RF_DNN_SVR_test_data'][f'{model.ECFP_Params}']
            recorder , model =model.get_DNN(train_set,test_set)
            return recorder
        
        elif isinstance(model,meta_models.meta_SVR):
            train_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
            test_set = data_set_dic['RF_DNN_SVR_test_data'][f'{model.ECFP_Params}']
            recorder , model =model.get_SVR(train_set,test_set)
            return recorder
        elif isinstance(model,meta_models.meta_GBM):
            train_set = data_set_dic['RF_DNN_SVR_train_data'][f'{model.ECFP_Params}']
            test_set = data_set_dic['RF_DNN_SVR_test_data'][f'{model.ECFP_Params}']
            recorder , model =model.get_GBM(train_set,test_set)
            return recorder
        
        

class nnls_SL():
    """NNLS为次级学习器的超级学习器程序
    能支持四种meta模型的输入
    先进行五折测试，再拟合次级模型，最后将NNLS的参数运用到验证集和训练集上
    """
    def __init__(self,path = None,train_df=None,test_df =None,_model_squential=None) -> None:
        self.fold_path = path
        self.train_df = train_df
        self.test_df = test_df
        self._model_squential = _model_squential
        print("将堆叠后的数据集放入self.train_nnls_on_XYdata即可训练第二层模型")
    
    def train_nnls(self, _model_squential: utils.model_sequential, 
                train_df: pd.DataFrame, 
                test_df: pd.DataFrame, 
                path='five_fold'):
        """
        训练 NNLS 超级学习器
        
        参数:
        - _model_squential: utils.model_sequential, 次级学习器模型
        - train_df: pd.DataFrame, 训练集数据
        - test_df: pd.DataFrame, 测试集数据
        - path: str, 保存五折测试结果的路径，默认为 'five_fold'
        """
        self.fold_path = path
        self.train_df = train_df
        self.test_df = test_df
        self._model_squential = _model_squential
        # 开始五折测试
        self.five_fold_test(_model_squential, train_df)
        
        df = self.get_fold_data()  # 获得堆叠后的测试集数据
        
        self.train_nnls_on_XYdata(df)
        
    def train_nnls_on_XYdata(self,df):
        # 使用堆叠后的测试集数据训练NNLS
        self.X = df.iloc[:, 1:]
        self.y = df['true']
        coefficients, residuals = nnls(self.X, self.y)
        self.coefficients = coefficients
        self.df = df
        print('The A of arrays is', coefficients)
        
    
    def VALI_test(self):
        """
        在验证集上测试模型
        返回:
        - SL_VALI_out: pd.DataFrame, 超级学习器在验证集上的输出结果
        """
        train_data = self.train_df
        VALI_data = self.test_df
        
        _path = f'VALI_test'
        eb_model = ensemble_models_new(model_save_files=_path)
        eb_model.train_meta_models(train_data,
                                    VALI_data,
                                    model_squential=self._model_squential,
                                    save=True)
        VALI_out_df =  pd.read_csv(f'{_path}/L1_test_data.csv', index_col=0)
        SL_VALI_out = self.get_all_df(VALI_out_df)  # 输出所有预测结果
        return SL_VALI_out
    
    def five_fold_test(self,
                    _model_squential: utils.model_sequential, 
                    train_df: pd.DataFrame,
                    ) -> pd.DataFrame:
        """
        进行五折测试
        
        参数:
        - _model_squential: utils.model_sequential, 学习器模型
        - train_df: pd.DataFrame, 训练集数据
        
        返回:
        无
        """

        data_loader = db()  # 请确保 db 已经定义并导入
        data_loader.data = train_df
        for i, (train, test) in enumerate(data_loader.get_folds()):
            _path = f'{self.fold_path}/fold_{i}'
            if os.path.exists(_path+"/L1_test_data.csv"):
                continue
            eb_model = ensemble_models_new(model_save_files=_path)
            eb_model.train_meta_models(train,
                                        test,
                                        model_squential=_model_squential,
                                        save=False)

        
        return 
    
    def get_fold_data(self):
        """
        获取堆叠后的测试集数据
        
        返回:
        - df_: pd.DataFrame, 堆叠后的测试集数据
        """
        for i in range(5):
            _path = f'{self.fold_path}/fold_{i}'
            _df = pd.read_csv(f'{_path}/L1_test_data.csv', index_col=0)
            if i == 0:
                df_ = _df
            else:
                df_ = pd.concat([df_, _df], axis=0)
        
        return df_   
        
    def get_all_df(self, df_meta):
        """
        使用 train_nnls 的系数用于加权 meta 的结果
        
        参数:
        - df_meta: pd.DataFrame, meta 模型的输出结果
        
        返回:
        - df_meta: pd.DataFrame, 加权后的 meta 模型输出结果
        """
        self._X = df_meta.iloc[:, 1:]
        self._y = df_meta['true']
        df_r = self._X * self.coefficients
        lis = []
        for i in range(len(df_r.index)):
            lis.append(df_r.iloc[i, :].sum())
        arr = np.array(lis)
        df_meta['SL'] = arr

        return df_meta
    
    def get_metric(self, df_meta):
        """
        获取模型的性能指标
        
        参数:
        - df_meta: pd.DataFrame, meta 模型的输出结果
        
        返回:
        - metrics: dict, 模型性能指标
        """
        _df = self.get_all_df(df_meta)
        return utils.cal_df(_df)
