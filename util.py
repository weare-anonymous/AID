from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import dill
from make_synthetic import generate_data

## ---------------------------- load data ----------------------------

def load_data(datatype, seed):
    if datatype in ['XOR', 'orange_skin', 'nonlinear_additive', 'fusion_feature', 'fusion_feature_new']:
        return load_data_synthetic(datatype, seed)
    elif datatype == 'diabetes':
        return load_data_diabetes(seed)
    elif datatype == 'mimic':
        return load_data_mimic(seed)
    else:
        print('not supported!')
        return

def load_data_synthetic(datatype, seed):
    """
    'XOR', 'orange_skin', 'nonlinear_additive': 
    (9000, 10) (1000, 10) (9000, 1) (1000, 1)
    
    'fusion_feature': 
    (9000, 10) (1000, 10) (9000, 3) (1000, 3)
    """
    x_train, x_val, y_train, y_val = generate_data(datatype=datatype, seed=seed)
    return x_train, x_val, y_train, y_val

def load_data_diabetes(seed):
    """
    (91589, 110) (10177, 110) (91589, 10) (10177, 10)
    """
    with open('data/diabetic.pkl', 'rb') as fin:
        res = dill.load(fin)
    data = res['data']
    label = res['label']
    selected_disease = res['selected_disease_ICD9']
    x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.1, random_state=seed)
    return x_train, x_val, y_train, y_val

def load_data_mimic(seed):
    """
    (41789, 1000), (4644, 1000), (41789, 11), (4644, 11)
    """
    with open('data/mimic.pkl', 'rb') as fin:
        res = dill.load(fin)
    x = res['x']
    y_disease = res['y_disease']
    y_mortality = res['y_mortality']
    y = np.concatenate([y_disease, y_mortality], axis=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=seed)
    return x_train, x_val, y_train, y_val

## ---------------------------- evaluation ----------------------------

def my_eval_rank(score):
    """
    input: score vector
    output: rank vector, the ranks are permutated for weights with the same score
    """
    score = abs(score)
    idx = np.random.permutation(len(score)) 
    permutated_weights = score[idx]  
    permutated_rank=(-permutated_weights).argsort().argsort()+1
    rank = permutated_rank[np.argsort(idx)]
    return np.array(rank)

def my_eval(y_true, y_pred_proba, y_blackbox_proba=None, verbose=False):
    """
    y_true, y_pred_proba, y_blackbox_proba are one-hot encoded
    (n_samples, n_targets)
    """
    if y_blackbox_proba is None:
        y_blackbox_proba = y_true
    y_blackbox = np.array(y_blackbox_proba > 0.5, dtype=np.float32)
    y_pred = np.array(y_pred_proba > 0.5, dtype=np.float32)
    n_class = y_true.shape[1]
    ret = {}
    for i in range(n_class):
        ## AUROC & AUPRC
        try:
            tmp_auroc = roc_auc_score(y_true[:,i], y_pred_proba[:,i])
        except:
            tmp_auroc = 0.0
        try:
            tmp_auprc = average_precision_score(y_true[:,i], y_pred_proba[:,i])
        except:
            tmp_auprc = 0.0
        
        ## Fidelity
        fidelity = accuracy_score(y_blackbox[:,i], y_pred[:,i])
        
        ## Accuracy
        accuracy = accuracy_score(y_true[:,i], y_pred[:,i])
            
        ret[i] = [tmp_auroc, tmp_auprc, fidelity, accuracy]
    ret_mat = []
    for k, v in ret.items():
        ret_mat.append(v)
    ret_mat = np.array(ret_mat)
    print('mean ROC-AUC: {0:.4f}, mean PR-AUC: {1:.4f}, mean fidelity: {2:.4f}, mean accuracy: {3:.4f}'.format(np.mean(ret_mat[:,0]), np.mean(ret_mat[:,1]), np.mean(ret_mat[:,2]), np.mean(ret_mat[:,3])))
    return ret






