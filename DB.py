import pandas as pd
from sklearn.model_selection import KFold
from datetime import datetime
from numpy import nan as NA
import numpy as np
from urllib.request import urlopen

class db:
    """数据加载器，衔接用户输入形式和内部运行数据格式
    输入形式应该为两列：第一列为smiles描述符或者casrn描述符，需要按照列名进行区分。第二列为毒性数据，需要带有相应列名
    若为casrn，需要先转换为smiles再进入后续计算，其中无法转换的数据，传入另一张df中
    """
    def __init__(self, file_path=None,target_name = "LD"):
        self.file_path = file_path
        self.data = pd.DataFrame([])
        self.transform_cas = False
        self.col_origin = []
        self.target_name = target_name
        if file_path !=None:
            self.data = self.read_data(self.file_path)
        
    def read_data(self, file_path) -> pd.DataFrame:
        """从文件路径读取数据，并提取特定列。

        Args:
            file_path (str): 数据文件的路径。

        Returns:
            pd.DataFrame: 包含'id', 'smiles', 'LogLD'列的DataFrame。
        """
        try:
            # 读取文件
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading the file: {e}")
            return pd.DataFrame([])  # 发生错误时返回空DataFrame

        # 删除任何含有缺失值的行，确保数据的完整性
        df.dropna(subset=['ids', 'smiles', self.target_name], inplace=True)
        # 检查是否需要将CAS转换为SMILES
        if 'smiles' not in df.columns:
            print("The 'smiles' column is not present. Attempting conversion from CAS.")
            df, self.loss_df = self.CAS2Smiles(df)
            self.transform_cas = True

        # 仅保留需要的列
        try:
            df = df[['ids', 'smiles', self.target_name]]
        except KeyError as e:
            print(f"Error: Required columns are missing from the data: {e}")
            return pd.DataFrame([])
        if self.target_name:
            df.rename(columns={self.target_name:"LogLD"},inplace=True)

        print(f"Data processing complete. Processed data size: {df.shape[0]}")
        return df


    def get_data(self):
        self.file_path = "pythonProject\DataBase\DB.csv"
        self.data = self.read_data(self.file_path)

    def search(self, x):
        if isinstance(self.data,pd.DataFrame):
            df = self.data
            location = df.index[df.isin(x).any(axis=1)]
            subdf = df.loc[location] 
            return [location, subdf]
        else:
            pass
    
    def CAS2Smiles(self,df : pd.DataFrame):
        """
        将表格中的第一列数据转化为smiles，返回完成的df，以及缺失值的df_loss
        Returns:
            _type_: _description_
        """
        """处理df，将casrn列转化为smiles

        Args:
            df (_type_): _description_
        """
        print('开始进行转化，共{}条数据'.format(df.shape[0]))
        
        smiles_lis =[]
        fine_lis=[]

        dic_output = {'smiles':[],f'{self.col_origin[1]}':[]}
        dic_loss = {'id':[],'cas':[],f'{self.col_origin[1]}':[]}
        
        for i in range(df.shape[0]):
            x = df.iloc[i,0]
            y = df.iloc[i,1]
            smi = self.CIRconvert(x)
            if smi !='Did not work':
                dic_output['smiles'].append(smi)
                dic_output[f'{self.col_origin[1]}'].append(y)
                
                print(i)
                smiles_lis.append(smi)
                fine_lis.append(i)
            else:
                print('cas: {} 未查询,返回NA'.format(x))
                dic_loss['cas'].append(x)
                dic_loss['id'].append(i)
                dic_loss[f'{self.col_origin[1]}'].append(y)
        return pd.DataFrame(dic_output) , pd.DataFrame(dic_loss)
    
    @staticmethod
    def CIRconvert(ids):
        """对于单个cas转化的函数

        Args:
            ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            try:
                ans = cirpy.resolve(str(ids) , 'smiles')
                if ans != None:
                    return ans
                else:
                    return 'Did not work'
            except:
                return 'Did not work'
            
        except:
            url = 'http://cactus.nci.nih.gov/chemical/structure/' + ids + '/smiles'
            ans = urlopen(url).read().decode('utf8')
            return ans
            
    @staticmethod
    def merge_and_average(a, b):
        # 对a的第二列进行修改
        print(a.shape)
        print(b.shape)
        b = b.set_index(a.index)

        a.iloc[:, 1] = (a.iloc[:, 1] + b.iloc[:, 1]) / 2.0

        # 将合并后的数据添加到a中，并返回
        return a

    def add_fuc(self, new_df, how="average"):
        # 检查输入是否为三列的数据框
        if len(new_df.columns) != 3:
            print("Error: Input dataframe should have 3 columns")
            return None

        # 检查输入数据框的smiles列中是否有与数据集相同的元素
        matching_rows = self.data.iloc[:, 2].isin(new_df.iloc[0:, 2])  # type: ignore 
        matching_rows2 = new_df.iloc[:, 2].isin(self.data.iloc[0:, 2]) # type: ignore 
        
        matching_df = self.data[matching_rows]
        self.data = self.data[~matching_rows]

        overlap = len(matching_df)

        if overlap > 0:
            print(f"{overlap} records with overlapping values found in dataset")

            # 先删除重合元素

            matching_df2 = new_df[matching_rows2]
            new_df = new_df[~matching_rows2]

            # 将融合后的记录添加到数据集中
            self.data.update(new_df)

            # 合并匹配的记录和新的记录
            ddf = self.merge_and_average(matching_df, matching_df2)
            # print(ddf)
            self.data = pd.concat([self.data, ddf])

            # 打印添加记录的数量
            print(f"{len(new_df)} records added to dataset")
            print(f"{len(matching_df)} records refresh to dataset")
        else:
            # 将新的记录添加到数据集中
            self.data = pd.concat([self.data, new_df])
            print(f"{len(new_df)} records added to dataset")
        self.data = self.data.reset_index(drop=True)

    def get_folds(self, df_data_p=None, fold=5, save_splits=False):
        """折叠切分数据集函数,返回迭代器
           df_data_p可以外接输入，或者默认加载过的数据
           默认5折cv
           save_splits 是否保存切割结果
        """
        df_data_p = self.data  # 得先加载数据再进行切割
        
        kf = KFold(n_splits=fold, shuffle=True, random_state=20)  # 设置折数
        for i, (train_index, test_index) in enumerate(kf.split(df_data_p)):
            print('fold{}'.format(i))
            train = df_data_p.iloc[train_index]
            test = df_data_p.iloc[test_index]
            if save_splits:
                train.to_csv('train_fold{}.csv'.format(i))
                test.to_csv('test_fold{}.csv'.format(i))
            yield (train, test)

    def save_data(self, save_path=None):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime('%d-%H-%M')

        # 将当前时间添加到文件名中
        filename = f"dataframe_refresh_{current_time}.csv"

        self.data.to_csv('../DataBase/{}'.format(filename))


dataloader = db