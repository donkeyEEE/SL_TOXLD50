import numpy as np
# from sklearn.decomposition import PCA, NMF
import pandas as pd
from deepchem.data import NumpyDataset,Dataset
import deepchem.feat
import os
from typing import Dict,List,Tuple


class DataLoader():
    # 用于加载数据，并且转化描述符的类
    def __init__(self,
                data_path:str=None,
                data_set:pd.DataFrame=None,
                ECFP_Params:list = [2048,2],
                Descriptor_type:str= 'ECFP',
                x_name = 'Smiles',
                y_name = 'LogLD'
                ) -> None:
        
        Descriptors = ['maccs','pubchem','rdkit','mordred']
        if Descriptor_type in Descriptors:
            if Descriptor_type == 'maccs':
                self.featurizer = deepchem.feat.MACCSKeysFingerprint()
            if Descriptor_type == 'pubchem':
                self.featurizer = deepchem.feat.PubChemFingerprint()
            if Descriptor_type == 'rdkit':
                self.featurizer = deepchem.feat.RDKitDescriptors()
            if Descriptor_type == 'mordred':
                self.featurizer = deepchem.feat.MordredDescriptors()
        else:
            self.featurizer = deepchem.feat.CircularFingerprint(size=ECFP_Params[0], radius=ECFP_Params[1])
        
        if data_path:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f'{data_path} is not exist')
            self.data_path = data_path
            df = self.read_data() # 获取原始数据
        else:
            df = data_set.copy()
        self.smiles = df[x_name].to_list()
        self.y_data = df[y_name].to_list()
        if len(self.smiles) != len(self.y_data):
            raise ValueError(f'chem num {len(self.smiles)} \
                                but lable num is {len(self.y_data)}')
        self.dataset = None
        
        
    def read_data(self)->pd.DataFrame:
        try:
            # 读取文件
            df = pd.read_csv(self.data_path)
            # 检测缺失值
            if df.isnull().values.any():
                print("DataFrame中存在缺失值")
                missing_rows = df[df.isnull().any(axis=1)]
                raise ValueError(f'{self.data_path} 中存在缺失值，位于：\n {missing_rows}')
        except Exception as e:
            raise ValueError(f"Error reading the file: {e}")
        return df
    
    def get_featurizer_data(self)->NumpyDataset:
        if self.dataset:
            return self.dataset
        data = self.featurizer.featurize(self.smiles)
        data = self.clean_array(data) # 避免出现不能计算的数据
        data = np.vstack(data) 
        self.dataset =NumpyDataset(X=data,y=self.y_data)
        return self.dataset
    
    def get_fold_data(self,seed=5233,ECFP_Params:list = [2048,2])->List[Tuple[Dataset, Dataset]]:
        from deepchem.splits import RandomSplitter
        import os
        splitter = RandomSplitter()

        splited_dataset = splitter.k_fold_split(dataset=self.get_featurizer_data(),
                                                k=5,
                                                seed=seed,
                                                )

        return splited_dataset

    @staticmethod
    def clean_array(arr):
        """
        清理数组中的NaN、无穷大和过大的数值。

        参数:
        arr (np.array): 需要清理的NumPy数组。

        返回:
        np.array: 清理后的数组。
        """
        # 检查并替换NaN值
        if np.isnan(arr).any():
            #print("发现NaN值，将它们替换为0.0")
            arr = np.nan_to_num(arr, nan=0.0)

        # 检查并替换无穷大的数值
        if np.isinf(arr).any():
            #print("发现无穷大值，将它们限制在-1e37到1e37之间")
            arr = np.clip(arr, -1e37, 1e37)

        # 检查并替换过大的数值
        if (arr > 1e10).any() or (arr < -1e10).any():
            #print("发现过大或过小的数值，将它们限制在-1e37到1e37之间")
            arr = np.clip(arr, -1e10, 1e10)

        #print("清理后的数组:", arr)
        return arr


