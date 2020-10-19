from matplotlib import pyplot as plt
import dill
import numpy as np
import pandas as pd

pd.set_option('precision',3)

def get_res(res, col, n_samples):
    all_mat = {}
    all_res = []
    for tmp_res in res:
        for k, v in tmp_res.items():
            if k in all_mat:
                all_mat[k].append(v)
            else:
                all_mat[k] = [v]
    for k, v in all_mat.items():
        all_mat[k] = np.array(v)
        best_idx = np.argmax(all_mat[k][:int(n_samples/100),col])
        all_res.append(all_mat[k][best_idx, col])
    return np.mean(all_res)

def get_res_generalizability(res, n_samples):
    res = np.array(res)
    if len(res.shape) == 2:
        res_mean = np.mean(np.array(res)[::int(n_samples/100)], axis=1)
    else:
        res_mean = res
    best_idx = np.argmax(res_mean)
    return res_mean[best_idx]
    
# ------------------------------ AUROC, AUPRC, fidelity, accuracy  ------------------------------

def stat_col(res_all, col):
    
    n_samples = 500

    dataset_list = ['XOR', 'nonlinear_additive', 'fusion_feature_new']
    method_list = ['AID']

    res_df = []
    for dataset in dataset_list:
        tmp_df = []
        for method in method_list:
            tmp_df.append(get_res(res_all[dataset][method]['res_val'], col, n_samples))
        res_df.append(tmp_df)
    res_df = pd.DataFrame(np.array(res_df).T, columns=dataset_list, index=method_list)
    return res_df

def stat_col_AID(res_all, n_samples, col):
    
    dataset_list = ['XOR', 'nonlinear_additive', 'fusion_feature_new']
    method_list = ['AID']

    res_df = []
    for dataset in dataset_list:
        tmp_df = []
        for method in method_list:
            tmp_df.append(get_res(res_all[dataset][method]['res_val'], col, n_samples))
        res_df.append(tmp_df)
    res_df = pd.DataFrame(np.array(res_df).T, columns=dataset_list, index=method_list)
    return res_df
        
    
# ## ------------------------------  generalizability ------------------------------

def stat_gen(res_all):
    
    n_samples = 500

    dataset_list = ['XOR', 'nonlinear_additive', 'fusion_feature_new']
    method_list = ['AID']

    res_df = []
    for dataset in dataset_list:
        tmp_df = []
        for method in method_list:
            tmp_df.append(get_res_generalizability(res_all[dataset][method]['res_generalizability'], n_samples))
        res_df.append(tmp_df)
    res_df = pd.DataFrame(np.array(res_df).T, columns=dataset_list, index=method_list)
    return res_df

def print_avg_latex(res_list):
    p = pd.Panel({n: df for n, df in enumerate(res_list)})
    df1 = p.mean(axis=0).round(3).astype(str)
    df2 = p.std(axis=0).round(3).astype(str)
    df_out = df1 + '#' + df2
    print(df_out.to_latex())
        
        
        
        
def stat_res(fname):
    
    roc_df_list = []
    pr_df_list = []
    fidelity_df_list = []
    accuracy_df_list = []
    gen_df_list = []

    with open(fname, 'rb') as fin:
        res_all_all = dill.load(fin)
                    
        for i_run in range(10):

            res_all = res_all_all[i_run]

            roc_df_list.append(stat_col(res_all, 2))
            pr_df_list.append(stat_col(res_all, 3))
            fidelity_df_list.append(stat_col(res_all, 0))
            accuracy_df_list.append(stat_col(res_all, 1))
            gen_df_list.append(stat_gen(res_all))
                
        
        print('fidelity_df_list')
        print_avg_latex(fidelity_df_list)
        print('roc_df_list')
        print_avg_latex(roc_df_list)
        print('pr_df_list')
        print_avg_latex(pr_df_list)
        print('gen_df_list')
        print_avg_latex(gen_df_list)
        print('accuracy_df_list')
        print_avg_latex(accuracy_df_list)
                
if __name__ == '__main__':
    stat_res('res/fill_your_pkl_name.pkl')
        
        
        
        