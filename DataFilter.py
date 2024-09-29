from models.dataloader import DataLoader
from models.meta_models import meta_SVR
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

class DataFilter():
    # 训练集筛选器
    # 基于SVR
    def __init__(self,csv_save_path:str = 'tmp.csv') -> None: #
        if os.path.exists(csv_save_path):
            self.df = pd.read_csv(csv_save_path)
        else:
            self.df = None
        self.path = csv_save_path
    def dfing(self,df:pd.DataFrame)->pd.DataFrame:
        if self.df:
            raise ValueError(f'No need for dfing, as {self.path} is already exist')
        # 记录的df
        _df = pd.DataFrame([])
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=20)  # 设置折数
        for i, (train_index, valid_index) in tqdm(enumerate(kf.split(df))):
            SVR =meta_SVR(ECFP_Params=[2048,2],C=3)
            TD = DataLoader(data_set = df.iloc[train_index])
            VD = DataLoader(data_set = df.iloc[valid_index])
            r_dic = SVR.run_model(train_set=TD.get_featurizer_data(),test_set=VD.get_featurizer_data())
            v_df = df.iloc[valid_index]
            v_df['RMSE_loss'] = DataFilter.Error_loss(r_dic['test_true'],r_dic['test_pre'])
            v_df['pre_LogLD'] = r_dic['test_pre']
            _df = pd.concat([_df,v_df],axis=0)
        self.df = _df
        return _df
    
    def load_get_data(self,thresholde):
        if self.df is not None:
            return self.filter_df(self.df,thresholde)
        else:
            self.df = pd.read_csv(self.path,index_col=0)
            return self.filter_df(self.df,thresholde)
    @staticmethod
    def filter_df(df:pd.DataFrame,thresholde)->pd.DataFrame:
        # 使用阈值对训练集进行筛选
        df_2 = df[df['RMSE_loss']<thresholde]
        print(f"去除误差在{thresholde}以上的化合物{df.shape[0]-df_2.shape[0]}，剩余{df_2.shape[0]}")
        return df_2

    @staticmethod
    def Error_loss(a:list,b:list):
        a = np.array(a)
        b = np.array(b)
        return np.sqrt((a-b)**2)