# para_dic.py
# 这个模块定义了用于模型训练的超参数字典，并实现了一个性能记录器类来存储和计算性能指标。
# 此外，它还提供了一个超参数记录器类来保存和导出超参数及其对应性能指标的记录。

# 超参数字典，定义了用于随机森林模型的参数范围

# 导入自定义的模型和工具模块
import models.utils as utils
import json

def cartesian_product(*lists):
    if not lists:
        yield []
        return
    for prefix in lists[0]:
        for suffix in cartesian_product(*lists[1:]):
            yield [prefix] + suffix

def get_para(para_dic):
    key_lis = list(para_dic.keys())
    lists = [para_dic[key] for key in key_lis]
    for product in cartesian_product(*lists):
        yield product



# 性能记录器类，用于记录和计算模型的性能指标
class PerformanceRecorder:
    def __init__(self, param: list):
        """
        初始化性能记录器。
        :param param: list, 模型训练的超参数列表
        """
        self.param_ = param  # 超参数
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
    def __init__(self,save_path=None):
        self.save_path = save_path
        self.records = HyperparameterRecorder.read_json(self.save_path)
    def add_record(self,  recorder:PerformanceRecorder,parameters=None):
        # 将PerformanceRecorder实例转换为字典
        recorder_dict = {
            'metric_average_train': recorder.metric_average_train,
            'metric_average_test': recorder.metric_average_test,
            'parameters': recorder.param_,
            'metric_train': recorder.metric_train,
            'metric_test': recorder.metric_test,
        }
        self.records[recorder.param_] = recorder_dict
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