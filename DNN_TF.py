import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import json
class DNN(keras.Model):
    def __init__(self, n_layers=1, layer_size=16, batch_size=256, learning_rate=0.0001, epochs=500, seed=9700, layer_size_lis=None, **kwargs):
        super().__init__(**kwargs)
        self._n_layers = n_layers
        self._layer_size = layer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.layer_size_lis = layer_size_lis
        self.generate_fcn()

    def generate_fcn(self):
        self.pipeline = []
        if self.layer_size_lis is None:
            for i in range(self.n_layers):
                self.pipeline.append(layers.BatchNormalization())
                self.pipeline.append(layers.Dense(self.layer_size, activation='relu'))
        else:
            for layer_size in self.layer_size_lis:
                self.pipeline.append(layers.BatchNormalization())
                self.pipeline.append(layers.Dense(layer_size, activation='relu'))
        self.pipeline.append(layers.BatchNormalization())
        self.pipeline.append(layers.Dense(1, activation='linear'))

    @property
    def n_layers(self):
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value):
        self._n_layers = value
        self.generate_fcn()

    @property
    def layer_size(self):
        return self._layer_size

    @layer_size.setter
    def layer_size(self, value):
        self._layer_size = value
        self.generate_fcn()

    def call(self, input_1, training=False):
        x = input_1
        for layer in self.pipeline:
            x = layer(x, training=training)
        return x

    def fit(self, x_train, y_train, **kwargs):
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.compile(optimizer=adam, loss='mse', metrics=['mae'])
        self.history = super().fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, **kwargs)
        

    def save_model(self, file_path):  # 'mp/model_0'
        tf.saved_model.save(self, file_path)

    @staticmethod
    def load_model(file_path):
        model = tf.keras.models.load_model(file_path)
        return model

    def get_config(self):
        return {
            "n_layers": self.n_layers,
            "layer_size": self.layer_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "seed": self.seed,
            "layer_size_lis": self.layer_size_lis
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_DNN(num_layers=3,
            ECFP_Params=[2048,2],
            layer_size_lis = None,
            learning_rate = 0.001,
            batch_size = 256
            )->DNN:
    model = DNN(learning_rate = learning_rate,
                n_layers = num_layers,
                layer_size = ECFP_Params[0],
                epochs = 3000,
                layer_size_lis = layer_size_lis,
                batch_size = batch_size
                )
    
    return model

from typing import Tuple
def remake_DNN(train_set, test_set ,model)->Tuple[dict,DNN]:
    """训练DNN模型

    Args:
        train_set (dc.data.NumpyDataset): 分子描述符+标签的训练集
        test_set (dc.data.NumpyDataset): 分子描述符+标签的测试集
        model (_type_): dc.models.KerasModel.

    Returns:
        _type_: fold_record,model
    """
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10,
                                            monitor='val_loss',
                                            mode='min',
                                            restore_best_weights=True)
            
    model.fit(train_set.X.astype(np.float32), train_set.y,
            callbacks=[early_stopping],
            validation_data=(test_set.X.astype(np.float32), test_set.y),
            #verbose=1
            ) ###
    
    y_hat = model.predict(test_set.X.astype(np.float32)).reshape(-1)
    y_hat_train = model.predict(train_set.X.astype(np.float32)).reshape(-1)
    
    fold_record = {'train_true': train_set.y, 'train_pre': y_hat_train, 'test_true': test_set.y, 'test_pre': y_hat}
    return fold_record,model