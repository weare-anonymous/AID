import os
import dill
from time import gmtime, strftime
from AID import run_AID

def save_pkl(res, fname):
    with open(fname, 'wb') as fout:
        dill.dump(res, fout)

if __name__ == '__main__':
    
    try:
        os.stat('models')
    except:
        os.mkdir('models')
    try:
        os.stat('res')
    except:
        os.mkdir('res')
    
    ## ==================================== exp settings ====================================
    
    datatype_list = ['XOR', 'nonlinear_additive', 'fusion_feature_new']
    fname = 'res/res_{0}.pkl'.format(strftime("%Y%m%d_%H%M%S", gmtime()))
    n_run = 10
    black_epochs = 20
    n_samples_list = [500]
    eval_more = True
    max_time = 600.0
    
    ## ==================================== run ====================================
    
    all_all_res = {}
    for i_run in range(n_run):
        print('='*30, i_run, '='*30)
    
        all_res = {}
        all_all_res[i_run] = all_res

        for datatype in datatype_list:
            print('='*30, datatype, '='*30)
            all_res[datatype] = {}

            if datatype in ['XOR']:
                num_features = 4
            elif datatype in ['orange_skin', 'nonlinear_additive']:
                num_features = 6
            elif datatype in ['fusion_feature', 'fusion_feature_new']:
                num_features = 8
            elif datatype in ['mimic']:
                num_features = 50

            ## --------------------- active ---------------------
            print('='*30, 'AID', '='*30)
            all_res[datatype]['AID'] = run_AID(datatype=datatype, 
                                                 black_epochs=black_epochs, 
                                                 num_features=num_features, 
                                                 solve_epochs=max(n_samples_list), 
                                                 mode='active', eval_more=eval_more,
                                                 seed=i_run)
            save_pkl(all_all_res, fname)
            
