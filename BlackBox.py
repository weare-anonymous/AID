from __future__ import print_function

import keras.backend as K
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential, Model
from keras.losses import binary_crossentropy
from keras.layers import Dense, Dropout, Activation, Input
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import regularizers


from make_synthetic import generate_data
import numpy as np
from util import *
import time 
import os
import dill
from time import gmtime, strftime



class BlackBox:
    """
    """
    def __init__(self, datatype, batch_size=32, epochs=10):
        # set parameters:
        self.batch_size = batch_size
        self.epochs = epochs
        run_id = strftime("%Y%m%d%H%M%S", gmtime())
        self.filepath="models/blackbox_model_{0}_{1}.hdf5".format(datatype, run_id)
        self.predpath="models/blackbox_preds_{0}_{1}.pkl".format(datatype, run_id)
        self.res = []
        self.datatype = datatype

    def load(self, x_train, x_val, y_train, y_val):
        print('Loading Data for Black-Box Model ...')
        self.x_train, self.x_val, self.y_train, self.y_val = x_train, x_val, y_train, y_val
        self.m_dim = self.x_train.shape[1]
        print('x_train shape:', self.x_train.shape)
        print('y_train shape:', self.y_train.shape)
        print('x_val shape:', self.x_val.shape)
        print('y_val shape:', self.y_val.shape)
        
    def build_model(self):
        model_input = Input(shape=(self.m_dim,), dtype='float32') 
        
        if self.datatype in ['XOR', 'orange_skin', 'nonlinear_additive']:
            net = Dense(100, activation='relu', name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3))(model_input)
            net = Dropout(0.5)(net)
            net = Dense(100, activation='sigmoid', name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3))(net)
            net = Dense(1, name = 'dense3')(net)
        elif self.datatype in ['fusion_feature']:
            net = Dense(100, activation='relu', name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3))(model_input)
            net = Dropout(0.5)(net)
            net = Dense(100, activation='sigmoid', name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3))(net)
            net = Dense(3, name = 'dense3')(net)
        elif self.datatype in ['fusion_feature_new']:
            net = Dense(100, activation='relu', name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3))(model_input)
            net = Dropout(0.5)(net)
            net = Dense(100, activation='sigmoid', name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3))(net)
            net = Dense(2, name = 'dense3')(net)
        elif self.datatype == 'mimic':
            net = Dense(256, activation='relu', name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3))(model_input)
            net = Dropout(0.5)(net)
            net = Dense(64, activation='sigmoid', name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3))(net)
            net = Dense(11, name = 'dense3')(net)
        elif self.datatype == 'LR':
            net = Dense(11, name = 'dense3')(model_input)
        elif self.datatype == 'diabetes':
            net = Dense(128, activation='relu', name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3))(model_input)
            net = Dropout(0.5)(net)
            net = Dense(128, activation='sigmoid', name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3))(net)
            net = Dense(10, name = 'dense3')(net)
        else:
            print('not supported!')
            
        net = Activation('sigmoid')(net)
        model = Model(model_input, net)
        return model

    def multitask_loss(self, y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))
    
    def train(self):
        print('Build Black-Box Model...')
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth=True   
        session = tf.Session(config=config)
        KTF.set_session(session)
        
        self.model = self.build_model()
        self.model.compile(loss=self.multitask_loss, optimizer='adam', metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(self.filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint] 
        st = time.time()
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), callbacks=callbacks_list, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        duration = time.time() - st
        print('Training time is {}'.format(duration))    
        
    def load_weights(self):
        self.load()
        print('Load Black-Box Model...')
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth=True   
        session = tf.Session(config=config)
        KTF.set_session(session)
        
        self.model = self.build_model()
        self.model.compile(loss=self.multitask_loss, optimizer='adam', metrics=['accuracy'])
                
    def gen_preds(self):
        self.model.load_weights(self.filepath, by_name=True) 
        pred_train = self.model.predict(self.x_train, verbose=0, batch_size = self.batch_size)
        pred_val = self.model.predict(self.x_val, verbose=0, batch_size = self.batch_size)
        self.pred_train = pred_train
        self.pred_val = pred_val
        my_eval(self.y_train, pred_train)
        self.res.append(my_eval(self.y_val, pred_val))
        res = {'pred_train':pred_train, 'pred_val':pred_val}
        with open(self.predpath, 'wb') as fout:
            dill.dump(res, fout)
        return pred_train, pred_val
    
    def gen_preds_new(self, x_train, x_val):
        self.model.load_weights(self.filepath, by_name=True) 
        pred_train = self.model.predict(x_train, verbose=0, batch_size = self.batch_size)
        pred_val = self.model.predict(x_val, verbose=0, batch_size = self.batch_size)
        res = {'pred_train':pred_train, 'pred_val':pred_val}
        with open(self.predpath, 'wb') as fout:
            dill.dump(res, fout)
        return pred_train, pred_val
    
    def predict_proba(self, xt, i_target):
        yt = self.model.predict(xt, batch_size = 64)
        tmp_proba = yt[:, i_target]
        yt = np.c_[1-tmp_proba, tmp_proba]
        return yt
    
    def predict(self, xt, i_target):
        yt = self.predict_proba(xt, i_target)
        return np.argmax(yt, axis=1)
    
if __name__ == '__main__':
    
    datatype = 'XOR'
    x_train, x_val, y_train, y_val = generate_data(n=1000, datatype=datatype)
    
    blackbox = BlackBox(datatype=datatype, epochs=20)
    blackbox.load(x_train, x_val, y_train, y_val)
    blackbox.train()
    blackbox.gen_preds()
#     with open('res/blackbox_res.pkl', 'wb') as fout:
#         dill.dump(blackbox.res, fout)    
    
    