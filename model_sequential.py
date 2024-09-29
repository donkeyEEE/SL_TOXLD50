from models.meta_models import meta_model,L2_model
from typing import Union
import os
import json
class ModelSequential:
    """定义集成模型结构
    传入模型定义类
    """
    def __init__(self, *args) -> None:
        self.functions:list[Union[meta_model,L2_model]] = list(args)
        self.set_models_id()
    
    def get_num_models(self):
        return len(self.functions)

    def set_models_id(self):
        for i,func in enumerate(self.functions):
            func.id = i
            
    def get_model_lis(self):
        return self.functions
    
    def get_model_names(self)->list:
        lis = []
        for _ in self.functions:
            lis.append(_.get_model_name())
        return lis
    def show_models(self):
        """依次调用self.function中的类属性.get_model_mark
        """
        print('The structure of Stacking model:')
        for func in self.functions:
            print(func.get_model_mark())

    def get_models_mark(self):
        """获取模型标记,返回字符串
        """
        _str = ''
        for func in self.functions:
            _str = _str + func.get_model_mark() +'\n'
        return _str
    
    def index_model(self,model_name:str)->meta_model:
        model_list = self.get_model_names()
        return self.functions[model_list.index(model_name)]
    
    def save(self,path:str='mp'):
        if os.path.exists(path) is not True:
            os.mkdir(path) # 创建MP
        dic = {}
        import json
        for _model in self.get_model_lis():
            _path = f'{path}/model_{_model.id}' # mp/model_0
            dic[_model.mark] = f'model_{_model.id}'
            _model.save(path=_path) 
        
        with open(f'{path}/model_squential.json', 'w') as json_file:
            json.dump(dic, json_file)
            
    @staticmethod
    def load(path:str='_All_Test\model_pretrained\metamodels'):
        with open(f'{path}/model_squential.json','rb') as file:
            dic = json.load(file)
        lis = []
        from models.meta_models import meta_model
        for _p in dic.values():
            lis.append(meta_model.load(path=f'{path}/{_p}'))
        return ModelSequential(*lis)