from models.SL import SuperLearner
from models.DataManager import DataManager_train,DataManager_valid,DataManager
import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class SL_evaluator():
    def __init__(self,SL:SuperLearner) -> None:
        self.SL = SL
        self.data = self.fast_gat_eval_data()
        self.dic = self.data['test_score'].to_dict()
        self.df_dic = self.organize_data(self.dic)
    
    def get_fold_test_metrics(self):
        return self.dic 
    
    def get_SL_metrics(self):
        return self.data['SL']
    def get_pres_on_trainset(self,rename=False):
        if rename:
            return self.rename_df(self.SL.valid_DataManager.df_labels_trainmodel)
        return self.SL.valid_DataManager.df_labels_trainmodel
    
    def get_train_data(self,rename=False):
        # 训练集上的预测和真实值
        if rename:
            return self.rename_df(self.SL.train_DataManager.df_stack_pres_and_labels)
        return self.SL.train_DataManager.df_stack_pres_and_labels
    
    def get_test_data(self,rename=False):
        if rename:
            return self.rename_df(self.SL.valid_DataManager.df_labels_testmode)
        return self.SL.valid_DataManager.df_labels_testmode
    
    def cal_AD(self,s=0.85,n=1):
        from models.AD_FP import ADFP
        from models import utils
        SL = self.SL
        tm = SL.train_DataManager
        vm = SL.valid_DataManager
        _df = vm.df_labels_testmode
        ADFPer = ADFP(train=tm.data_set,test=vm.data_set)
        ADFPer.cal_AD_by_FP(S=s,min_num=n)
        ori_df =  ADFPer.Using_AD(_df.copy(),if_get_metric=False)
        metric_df = utils.cal_df(ori_df,label_name='labels')
        metric_r2 = metric_df['L2_MLR']['R2']
        data = ADFPer.test
        ori_df['is_ADs'] = data['is_ADs']
        num_ADs =  sum(data['is_ADs']==1)
        return {'S':s,'metrics':metric_df,'R2':metric_r2,'data_set':ori_df,'num_ADs':num_ADs}
    @staticmethod
    def rename_df(df:pd.DataFrame):
        models = ['XGB', 'SVR', 'DNN', 'RF']
        fps = ['ECFP', 'Mordred', 'MACCS', 'Rdkit']
        # 生成16种模型选项
        model_options = [f"{model} {variant}" for model in models for variant in fps] +['L2_MLR']
        df.rename(dict(zip([i for i in range(16)],model_options)),axis=1,inplace=True)
        return df
    
    def plot_scatter(self,model = 'RF_ECFP'):
        tm:DataManager_train = self.SL.train_DataManager
        vm:DataManager_valid = self.SL.valid_DataManager
        #####
        models = ['XGB', 'SVR', 'DNN', 'RF']
        fps = ['ECFP', 'Mordred', 'MACCS', 'Rdkit']
        # 生成16种模型选项
        model_options = [f"{model}_{variant}" for model in models for variant in fps] +['L2_MLR']
        for i,_ in enumerate(model_options):
            if _ == model:
                break
        if i ==15 and model_options[-1] != model :
            raise ValueError(f'{model}')
        #####
        from models.utils import plot_parity_plus
        _df = vm.df_labels_testmode
        plot_parity_plus(y_true=_df['labels'],
                        y_pred=_df[i if i !=16 else 'L2_MLR'],
                        name=model,
                        )
    
    
    @staticmethod
    def organize_cross_validation_data(data):
        # Initialize an empty list to store rows of the DataFrame
        rows = []
        models = ['XGB', 'SVR', 'DNN', 'RF']
        fps = ['ECFP', 'Mordred', 'MACCS', 'Rdkit']
        # 生成16种模型选项
        model_options = [f"{model}_{variant}" for model in models for variant in fps]
        # Loop through the data
        for fold_idx, fold_data in enumerate(data):
            for model_idx, metrics in fold_data.items():
                for metric, value in metrics.items():
                    # Append each metric as a separate row
                    rows.append({
                        'Model': model_options[model_idx],
                        'Fold': fold_idx + 1,
                        'Metric': '$R^2$' if metric=='R2' else metric,
                        'Value': value
                    })

        # Create a DataFrame from the rows list
        df = pd.DataFrame(rows)
        
        return df

    def fast_gat_eval_data(self):
        SL = self.SL
        _dic = SL.eval()
        CV_score_A = _dic['TrainRecord'].models_performance_nfold_average
        test_score = _dic['ValidRecord'].models_performance_validset
        CV_score = self.organize_cross_validation_data(_dic['TrainRecord'].models_performance_nfold)
        df_CV = pd.DataFrame(CV_score_A)
        df_test = pd.DataFrame(test_score)
        
        _lis = [_.mark for _ in SL.model_sequential.functions]
        df_CV.columns = _lis
        df_test.columns = _lis
        
        #df_CV['L2'] = _dic['TrainRecord'].L2models_performance['pres_lables_L2_MLR'].values()
        L2 = _dic['ValidRecord'].L2models_performance_validset['L2_MLR']
        
        return {'CV_score_A':df_CV,'test_score':df_test,'CV_score':CV_score,'SL':L2}
    
    
    @staticmethod
    def organize_data(evaluation_data:dict):
        # 模型和指纹类型的映射
        model_mapping = {
            'Random Forest Regressor': 'RF',
            'Deep neural network': 'DNN',
            'Support vector regression': 'SVR',
            'XGB': 'XGB'
        }
        fingerprint_mapping = {
            'ECFP': 'ECFP',
            'maccs': 'maccs',
            'rdkit': 'rdkit',
            'mordred': 'mordred'
        }

        # 评价指标
        metrics = ['MAE', 'RMSE', 'R2']

        # 创建空的DataFrame用于存储每个评价指标的数据
        data_frames = {metric: pd.DataFrame(index=fingerprint_mapping.values(), columns=model_mapping.values()) for metric in metrics}

        # 填充DataFrame
        for key, values in evaluation_data.items():
            for raw_model, model in model_mapping.items():
                if raw_model in key:
                    for fp_key, fp in fingerprint_mapping.items():
                        if fp_key in key:
                            for metric in metrics:
                                if metric in values:
                                    data_frames[metric].loc[fp, model] = values[metric]
                            break

        return data_frames

    @staticmethod
    def rename_df(df:pd.DataFrame):
        models = ['XGB', 'SVR', 'DNN', 'RF']
        fps = ['ECFP', 'Mordred', 'MACCS', 'Rdkit']
        # 生成16种模型选项
        model_options = [f"{model} {variant}" for model in models for variant in fps] +['L2_MLR']
        df.rename(dict(zip([i for i in range(16)],model_options)),axis=1,inplace=True)
        return df

    def plot_heatmap(self,df=None,mode = 'test',metric='R2', cmap_style='viridis', font_size=12, label_font_size=14, annot_font_size=10, tick_font_size=12, save_path=None):
        """
        绘制单个评价指标的热力图。

        参数:
        df (DataFrame): 包含特定评价指标数据的DataFrame。
        metric (str): 要绘图的评价指标名称。
        cmap_style (str): 热力图的颜色映射风格，可选值包括 'viridis', 'plasma', 'inferno', 'magma',
                        'Blues', 'Greens', 'Reds', 'coolwarm', 'bwr', 'seismic', 'twilight', 'Pastel1', 'Set1', 'Set3' 等。
        font_size (int): 整体字体大小。
        label_font_size (int): XY轴标签字体大小。
        annot_font_size (int): 数字注释字体大小。
        tick_font_size (int): XY轴刻度字体大小。
        save_path (str): 如果指定，将图像保存到该路径。
        """
        if df is None:
            if mode == 'test':
                df = self.df_dic[metric]
            elif mode == 'CV':
                df = self.organize_data(self.data['CV_score_A'].to_dict())[metric]
            else:
                raise ValueError(f'{mode}')
        
        # 设置整体字体和大小
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = font_size

        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(df.astype(float), annot=True, fmt=".3f", cmap=cmap_style, linewidths=.5, annot_kws={"size": annot_font_size})
        plt.title(f'Heatmap of {metric}', fontsize=font_size + 4)
        plt.xlabel('Model', fontsize=label_font_size)
        plt.ylabel('Fingerprint', fontsize=label_font_size)
        
        # 设置XY轴刻度字体大小
        heatmap.tick_params(axis='x', labelsize=tick_font_size)
        heatmap.tick_params(axis='y', labelsize=tick_font_size)

        # 如果指定了保存路径，则保存图像
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        # 显示图形
        plt.show()

# 示例使用：
# plot_heatmap(data_frames['MAE'], 'MAE', cmap_style='plasma', font_size=12, label_font_size=14, annot_font_size=10, tick_font_size=12, save_path='heatmap_MAE.png')
