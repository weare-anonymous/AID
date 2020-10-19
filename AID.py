from __future__ import print_function

from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from keras.engine.topology import Layer 
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Model 
from keras.layers import Activation, Dense, Input
import numpy as np

import os
from util import *
from BlackBox import *
import dill
import time 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
import scipy as sc

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

np.random.seed(0)
tf.set_random_seed(0)

### -------------------------------- Explainer --------------------------------
class EmbeddingLayer(Layer):
    """
    """
    def __init__(self, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(EmbeddingLayer, self).build(input_shape)
    
    def call(self, inputs):
        return inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class MaskWeightLayer(Layer):
    def __init__(self, l, n_targets, **kwargs):
        self.l = l
        self.n_targets = n_targets
        self.eps = 1e-8
        super(MaskWeightLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # (none, m_dim, m_dim)
        self.w_att = self.add_weight(name='w_att', shape=(input_shape[1], input_shape[1]), initializer='uniform', trainable=True)
        # (none, m_dim, n_targets)
        self.w_b = self.add_weight(name='w_b', shape=(input_shape[1], self.n_targets), initializer='uniform', trainable=True)
        super(MaskWeightLayer, self).build(input_shape)
        
    def call(self, inputs):
        x = inputs
        
        ### scores_soft: (none, m)
        scores_soft = tf.einsum('km,ms->ks', x, self.w_att) # s == m
        scores_soft = tf.nn.softmax(scores_soft) + tf.constant(self.eps)
        
        ### scores_hard: (m, n)
        scores_hard = self.w_b
        threshold = tf.expand_dims(tf.nn.top_k(tf.einsum('mn->nm', scores_hard), self.l, sorted = True)[0][:,-1], -1)
        threshold = tf.einsum('nm->mn', threshold)
        scores_hard = scores_hard - threshold + self.eps
        
        # in train phase, use differentiable scores_hard_train to approximate sign operation
        scores_hard_train = (tf.maximum(-1.0, tf.minimum(1.0, scores_hard))+ 1)/2
        # in evaluation phase, use sign operation
        scores_hard_eval = (tf.sign(scores_hard) + 1)/2
        scores_hard = K.in_train_phase(scores_hard_train, scores_hard_eval) 
        
        ### multiply scores_soft and scores_hard
        mask_weight = tf.einsum('mn,km->kmn', scores_hard, scores_soft)
        out = tf.einsum('kmn,km->kmn', mask_weight, x)
        return [out, scores_hard, scores_soft, mask_weight]
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1], self.n_targets), (input_shape[1], self.n_targets), (input_shape[0], input_shape[1]), (input_shape[0], input_shape[1], self.n_targets)]

class PredictLayer(Layer):
    """
    """
    def __init__(self, **kwargs):
        super(PredictLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True)
        super(PredictLayer, self).build(input_shape)
    
    def call(self, inputs):
        logits = tf.einsum('kmn,mn->kn', inputs, self.w)
        return logits
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
class ExplainerNN:
    """
    """
    def __init__(self, m_dim, n_targets, l):
        self.m_dim = m_dim
        self.n_targets = n_targets
        self.l = l

    def build(self):
        
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth=True   
        session = tf.Session(config=config)
        KTF.set_session(session)

        print('Initializing Explainer ...')
        
        ### (none, m_dim) -> (none, m_dim)
        input_x = Input(shape=(self.m_dim,), dtype='float32', name='input_x')
        embed_x = EmbeddingLayer()(input_x)
        
        ### (none, m_dim) -> (none, m_dim, n_targets)
        weighted_embed_x, scores_hard, scores_soft, mask_weight = MaskWeightLayer(self.l, self.n_targets, name='mask_weight_layer')(embed_x)
        
        ### (none, m_dim, n_targets) -> (none, n_targets)
        logits = PredictLayer()(weighted_embed_x)
        
        self.model = Model(inputs=input_x, outputs=logits)
#         adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        
#         self.model_mask_weight = K.function([input_x], [scores_hard, scores_soft, mask_weight])
        self.model_mask_weight = Model(inputs=input_x, outputs=mask_weight)
                
    def update(self, x, r, epochs=1):
        self.model.fit(x, r, batch_size=128, epochs=epochs, verbose=0)
        
    def get_mask_weight(self, x):
#         scores_hard, scores_soft, mask_weight = self.model_mask_weight([x])
#         [scores_hard, scores_soft, mask_weight] = self.model_mask_weight.predict(x, batch_size=32)
        mask_weight = self.model_mask_weight.predict(x, batch_size=32)
        return mask_weight
    
    def predict(self, x):
        logits = self.model.predict(x)
        return logits
    
    
### -------------------------------- Solver --------------------------------

class SolveNN:
    
    
    def __init__(self, l, m_dim, n_targets, blackbox_pred_path, datatype, seed, eval_more, mode, epochs=50):
        self.mode = mode # active or passive
        self.datatype = datatype
        self.epochs = epochs+1
        self.predpath=blackbox_pred_path
        self.i_epoch = 0
        
        self.zeta = 0.01

        self.chosen_idx_set = set([])
        self.tmp_x_train = np.array([])
        
        self.m_dim = m_dim
        self.n_targets = n_targets
        self.l = l

        self.val_step = 100
        self.eval_val = True
        self.eval_train = False
        self.eval_rank = False
        self.eval_gen = eval_more
        self.all_val_rank = []
        self.all_scores = []
        self.res_val = []
        self.res_train = []
        self.n_generalizability = 100
        self.n_nearby = 100
        self.res_generalizability = []
        self.duration = []
        self.output_pred_list = []
        
        self.is_debug = False
        
        self.explainer = ExplainerNN(self.m_dim, self.n_targets, self.l)
        self.explainer.build()
        self.extra_save = 'res/extra_AID.pkl'
                
    def load(self, x_train, x_val, y_train, y_val):
        print('Loading Data ...')
        self.x_train, self.x_val, self.y_train, self.y_val = x_train, x_val, y_train, y_val
        print('x_train shape:', self.x_train.shape)
        print('y_train shape:', self.y_train.shape)
        print('x_val shape:', self.x_val.shape)
        print('y_val shape:', self.y_val.shape)
        
        with open(self.predpath, 'rb') as fin:
            res = dill.load(fin)
        self.blackbox_pred_train, self.blackbox_pred_val = res['pred_train'], res['pred_val']
        if self.is_debug:
            print(self.blackbox_pred_train.shape, self.blackbox_pred_val.shape)
        self.logits_train = self.phi_inverse(self.blackbox_pred_train)
    
        ## query nearest
        self.tree = KDTree(self.x_val, leaf_size=3)    
        
    def get_res(self):
        res = {}
        res['y_val'] = self.y_val
#         res['pred_val'] = self.output_pred_list
        res['blackbox_pred_val'] = self.blackbox_pred_val
        if self.eval_rank:
            res['rank'] = self.all_val_rank
        if self.eval_val:
            res['res_val'] = self.res_val
        if self.eval_train:
            res['res_train'] = self.res_train
        if self.eval_gen:
            res['res_generalizability'] = self.res_generalizability
        res['time'] = self.duration
#         res['scores'] = self.all_scores
        res['eval_blackbox'] = my_eval(self.y_val, self.blackbox_pred_val, self.blackbox_pred_val)
        res['pred_blackbox'] = self.blackbox_pred_val
        return res
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def phi_inverse(self, x):
        eps = 1e-8
        return np.log(x/(1-x+eps))
    
    def KL_dist(self, p_x, q_x):
        p_x = np.array(p_x)
        q_x = np.array(q_x)
        dist = np.sum(p_x*np.log(p_x/q_x))
        return dist
    
    def step(self):
        self.i_epoch += 1
        
        ### update
        if self.mode == 'passive':
            self.explainer.update(self.x_train, self.logits_train, epochs=10)
        elif self.mode == 'active':
            
            n_sample_tmp = self.tmp_x_train.shape[0]
            if n_sample_tmp > 1:
                ### use previously selected data to build a region recognition classifier
                
                ## step 1: for previous queried data, get probabilities of blackbox model and surrogate model
                n_sample_tmp = self.tmp_x_train.shape[0]
                self.pred_surrogate =  self.sigmoid(self.explainer.predict(self.tmp_x_train))
                self.pred_blackbox = self.blackbox_pred_train[list(self.chosen_idx_set)]
                ## step 2: compute label using given zeta
                v_KL_dist = np.vectorize(self.KL_dist)
                kl_dist = v_KL_dist(self.pred_surrogate, self.pred_blackbox)
                self.region_label = np.array(kl_dist > self.zeta, dtype=np.float32)
                ## step 3: build a classifier for disagreement region
                ## step 4: indicate training data whether in disagreement region
                clf = LR()
                try:
                    clf.fit(self.tmp_x_train, self.region_label)
                    region_pred_label = clf.predict(self.x_train)
                except ValueError:
                    region_pred_label = np.ones(self.x_train.shape[0])
            else:
                region_pred_label = np.ones(self.x_train.shape[0])
            
            self.pred_train =  self.sigmoid(self.explainer.predict(self.x_train))
            ### use entropy to select sample
            pred_train = np.stack([self.pred_train, 1-self.pred_train], axis=-1)
            tmp = np.reshape(pred_train, (-1, pred_train.shape[-1])).T
            tmp1 = sc.stats.entropy(tmp)
            tmp2 = np.reshape(tmp1, (pred_train.shape[0], pred_train.shape[1]))
            entropy_train = np.sum(tmp2, axis=1)
            entropy_train_order = np.argsort(entropy_train, kind='stable')[::-1]
            ## loop idx in desc order of entropy
            for idx in entropy_train_order:
                ## idx is not selected before, and is in disagreement region
                if (idx not in self.chosen_idx_set) and (region_pred_label[idx] == 1):
                    self.chosen_idx_set.add(idx)
                    if self.is_debug:
                        print(entropy_train[idx])
                    break
            if self.is_debug:
                print('choose: ', idx)
            tmp_x_train = self.x_train[list(self.chosen_idx_set)]
            self.tmp_x_train = tmp_x_train
            if self.is_debug:
                print('current train samples: ', tmp_x_train.shape)
            tmp_logits_train = self.logits_train[list(self.chosen_idx_set)]

            ### update explainer
            self.explainer.update(self.tmp_x_train, tmp_logits_train)
            
        
    def train(self):
        for self.i_epoch in range(self.epochs):
            self.step()
            
            yes_val = False
            if self.is_debug:
                if self.datatype == 'XOR' and np.sum(self.z[0][[0,1]]) == 2:
                    yes_val = True
                elif self.datatype == 'orange_skin' and np.sum(self.z[0][[2,3,4,5]]) == 4:
                    yes_val = True
                elif self.datatype == 'nonlinear_additive' and np.sum(self.z[0][[6,7,8,9]]) == 4:
                    yes_val = True
                else:
                    yes_val = False

            ## eval and log info
#             val_steps = [10, 20, 50, 100, 200, 500]
#             if (self.i_epoch in val_steps) or (self.mode=='passive'):
            if (self.i_epoch % self.val_step == 0) or (yes_val) or (self.mode=='passive'):
                
#                 K.set_learning_phase(0)
#             if (self.i_epoch % self.val_step == 0) or (yes_val) or (self.mode=='passive'):
        
                print('epoch {0}'.format(self.i_epoch))
                
                ### compute pred val
                if self.eval_val:
                    t1 = time.time()
                    n_sample_val = self.x_val.shape[0]
                    mask_weight = self.explainer.get_mask_weight(self.x_val)
                    self.pred = self.sigmoid(self.explainer.predict(self.x_val))
                    self.scores = mask_weight
                    self.all_scores.append(mask_weight)
                    t2 = time.time()
                    self.duration.append(t2 - t1)
                    
                    # auc, auprc, fidelity, accuracy
                    self.output_pred_list.append(self.pred)
                    self.res_val.append(my_eval(self.y_val, self.pred, self.blackbox_pred_val))
                    
                    if self.eval_gen:
                        # generalizability
                        tmp_generalizability = []
                        all_nearby_pred = []
                        all_nearby_gt = []
                        for i_generalizability in range(self.n_generalizability):
                            idx = np.random.choice(n_sample_val)
                            _, ind = self.tree.query(self.x_val[[idx]], k=self.n_nearby)
                            nearby_samples = self.x_val[ind[0]]
                            nearby_pred = self.sigmoid(self.explainer.predict(nearby_samples))
                            nearby_gt = self.y_val[ind[0]]
                            all_nearby_pred.append(nearby_pred)
                            all_nearby_gt.append(nearby_gt)
                        all_nearby_pred = np.concatenate(all_nearby_pred, axis=0)
                        all_nearby_gt = np.concatenate(all_nearby_gt, axis=0)
                        all_nearby_pred = np.array(all_nearby_pred > 0.5, dtype=np.float32)
                        for i_target in range(self.n_targets):
                            tmp_generalizability.append(accuracy_score(all_nearby_pred[:,i_target], all_nearby_gt[:,i_target]))
                        self.res_generalizability.append(np.array(tmp_generalizability))
                        print('generalizability: {0:.4f}'.format(np.mean(tmp_generalizability)))
                
                ### compute pred train
                if self.eval_train:
                    n_sample_train = self.x_train.shape[0]
                    self.pred_train = self.sigmoid(self.explainer.predict(self.x_train))
                    self.res_train.append(my_eval(self.y_train, self.pred_train, self.blackbox_pred_train))
        
                if self.eval_rank:
                    all_rank = []
                    self.all_val_rank.append(all_rank)
                    self.all_scores.append(self.scores)
                    for i in range(self.scores.shape[0]):
                        score = self.scores[i,:,0]
                        ranks = my_eval_rank(score)
                        all_rank.append(ranks)
                    print('rank: {0}'.format(np.mean(all_rank)))
                    
                res = self.get_res()
#                 with open(self.extra_save, 'wb') as fout:
#                     dill.dump(res, fout)
                    
#                 K.set_learning_phase(1)
        
### -------------------------------- Solver --------------------------------

def run_AID(datatype, black_epochs, num_features, solve_epochs, mode, seed, eval_more, n_samples=None):
    
    np.random.seed(seed)
    tf.set_random_seed(seed)

# if __name__ == '__main__':
#     seed = 0
#     datatype = 'mimic'
#     mode = 'passive'
#     black_epochs = 1
#     solve_epochs = 5
#     n_samples = 1000
#     num_features = 2
#     eval_more = False
        
    x_train, x_test, y_train, y_test = load_data(datatype=datatype, seed=seed)
    blackbox = BlackBox(datatype=datatype, epochs=black_epochs)
    blackbox.load(x_train, x_test, y_train, y_test)
    blackbox.train()
    
    if mode == 'passive':
        if n_samples > 0:
            shuffle_idx = np.random.permutation(y_train.shape[0])
            x_train = x_train[shuffle_idx[:n_samples]]
            y_train = y_train[shuffle_idx[:n_samples]]
    m_dim = x_test.shape[1]
    n_targets = y_test.shape[1]
    blackbox.gen_preds_new(x_train, x_test)

    opt = SolveNN(l=num_features, m_dim=m_dim, n_targets=n_targets, blackbox_pred_path=blackbox.predpath, epochs=solve_epochs, seed=seed, mode=mode, datatype=datatype, eval_more=eval_more)
    opt.load(x_train, x_test, y_train, y_test)
    opt.train()
    res = opt.get_res()
    return res