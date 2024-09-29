"""
AD_FP.py

该模块专注于使用分子指纹进行适用域（AD）分析。它通过计算化合物之间的Tanimoto相似度来评估预测的适用性，并提供指纹分析相关的工具。模块还实现了相似性矩阵的计算和适用域的管理。

主要功能：
- 从SMILES字符串提取分子指纹（如MACCS）。
- 计算指纹之间的Tanimoto相似度。
- 实现基于相似度指标的AD检查流程。
- 根据定义的AD标准过滤测试集。
"""

import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit import DataStructs
import pandas as pd
import models.utils as utils

class ADBase:
    """
    ADBase 超类，用于适用域分析的通用方法。
    子类可以根据不同的指纹类型来实现特定的指纹生成和相似度计算。
    """
    def __init__(self, train, test, X='Smiles', Y='LogLD', S=None, min_num=None) -> None:
        self.train = train
        self.test = test
        self.X = X
        self.Y = Y
        self.threshold = S
        self.min_num = min_num
        self.fp_df = None  # 指纹相似度数据框
        self.test_AD_process()

    def get_fingerprint(self, smiles):
        """
        抽象方法：获取指定化合物的分子指纹。
        需要在子类中实现不同的指纹类型生成逻辑。
        """
        raise NotImplementedError

    def get_similarity(self, fp1, fp2):
        """
        抽象方法：计算两个指纹之间的相似度。
        需要在子类中实现不同指纹类型的相似度计算。
        """
        raise NotImplementedError

    def test_AD_process(self, test=None):
        if test is not None:
            self.test = test
        test_fps = self.get_fingerprint(list(self.test[self.X]))
        train_fps = self.get_fingerprint(list(self.train[self.X]))
        
        similarity_dict = {}
        from tqdm import tqdm
        for i in tqdm(range(len(test_fps))):
            similarity_dict[i] = []
            for train_fp in train_fps:
                similarity = self.get_similarity(train_fp, test_fps[i])
                similarity_dict[i].append(similarity)
        
        df = pd.DataFrame(similarity_dict).T
        df.columns = self.train[self.X]
        df.index = self.test[self.X]
        self.fp_df = df
        return df

    def check_test(self, S, N):
        """
        检查测试集中化合物是否在适用域内。
        S: 相似性阈值
        N: 最小相似化合物数量
        """
        df = self.fp_df
        is_ADs = []
        for i in range(len(df.index)):
            test_row = df.iloc[i, :]
            num_similar = sum(test_row > S)
            is_ADs.append(1 if num_similar >= N else 0)
        self.test['is_ADs'] = is_ADs
        return self.test

    def cal_AD_by_FP(self, S=None, min_num=None):
        if S is not None:
            self.threshold = S
        if min_num is not None:
            self.min_num = min_num
        df = self.check_test(S=self.threshold, N=self.min_num)
        num_in_AD = df[df['is_ADs'] == 1].shape[0]
        print(f'使用参数 S={self.threshold}, min_num={self.min_num}')
        print(f'适用域内有 {num_in_AD} 个化合物')
        return df

    def Using_AD(self, _df: pd.DataFrame, label='labels', if_get_metric=True):
        df = self.test
        _df['is_ADs'] = df['is_ADs']
        self.num_chem_after_filtered = _df[_df['is_ADs'] == 1].drop(['is_ADs'], axis=1).shape[0]
        if if_get_metric:
            return utils.cal_df(_df[_df['is_ADs'] == 1].drop(['is_ADs'], axis=1), label_name=label)
        else:
            return _df[_df['is_ADs'] == 1].drop(['is_ADs'], axis=1)


from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ADECFPBase(ADBase):
    """
    ADECFPBase 类，继承自 ADBase，统一使用 ECFP 指纹进行适用域分析。
    """
    @staticmethod
    def get_fingerprint(smiles:list) ->list:
        """
        使用 ECFP 指纹提取方法。
        """
        from deepchem.feat import CircularFingerprint
        f = CircularFingerprint(radius=2,size=1024)
        a = f.featurize(smiles)
        return a.tolist()

class ADECFP_Euclidean(ADECFPBase):
    """
    ADECFP_Euclidean 类，继承自 ADECFPBase，用于基于 ECFP 指纹计算欧几里得距离的适用域分析。
    """
    
    @staticmethod
    def get_similarity(fp1, fp2):
        """
        使用欧几里得距离计算 ECFP 指纹的相似性。
        """
        if fp1 is None or fp2 is None:
            return 0.0
        fp1_array = np.array(fp1)
        fp2_array = np.array(fp2)
        return np.linalg.norm(fp1_array - fp2_array)

class ADECFP_Pearson(ADECFPBase):
    """
    ADECFP_Pearson 类，继承自 ADECFPBase，用于基于 ECFP 指纹计算皮尔逊相关系数的适用域分析。
    """
    
    @staticmethod
    def get_similarity(fp1, fp2):
        """
        使用皮尔逊相关系数计算 ECFP 指纹的相似性。
        """
        if fp1 is None or fp2 is None:
            return 0.0
        fp1_array = np.array(fp1)
        fp2_array = np.array(fp2)
        # 计算皮尔逊相关系数
        return np.corrcoef(fp1_array, fp2_array)[0, 1]
    
from rdkit.Chem import MACCSkeys
class ADECFP_Tanimoto(ADECFPBase):
    """
    ADECFP_Tanimoto 类，继承自 ADECFPBase，
    用于基于 MACCS 指纹计算谷本系数（Tanimoto 系数）的适用域分析。
    注意：无法使用ECFP指纹计算，故而使用谷本系数
    """
    @staticmethod
    def get_maccs_fingerprint_(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return maccs_fp
    
    @staticmethod
    def get_fingerprint(smiles_lis):
        lis = []
        for m in smiles_lis:
            fp = ADECFP_Tanimoto.get_maccs_fingerprint_(m)
            lis.append(fp)
        return lis
    @staticmethod
    def get_similarity(fp1, fp2):
        """
        使用谷本系数（Tanimoto 系数）计算 ECFP 指纹的相似性。
        """
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)


class ADECFP_Cosine(ADECFPBase):
    """
    ADECFP_Cosine 类，继承自 ADECFPBase，用于基于 ECFP 指纹计算余弦相似度的适用域分析。
    """
    
    @staticmethod
    def get_similarity(fp1, fp2):
        """
        使用余弦相似度计算 ECFP 指纹的相似性。
        """
        if fp1 is None or fp2 is None:
            return 0.0
        fp1_array = np.array(fp1)
        fp2_array = np.array(fp2)
        return cosine_similarity([fp1_array], [fp2_array])[0][0]

class ADECFP_Hamming(ADECFPBase):
    """
    ADECFP_Hamming 类，继承自 ADECFPBase，用于基于 ECFP 指纹计算汉明距离的适用域分析。
    """
    
    @staticmethod
    def get_similarity(fp1, fp2):
        """
        使用汉明距离计算 ECFP 指纹的相似性。
        """
        if fp1 is None or fp2 is None:
            return 0.0
        fp1_array = np.array(fp1)
        fp2_array = np.array(fp2)
        return np.sum(fp1_array != fp2_array) / len(fp1_array)
