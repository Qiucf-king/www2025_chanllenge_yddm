import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import multiprocessing as mp
import gc
from scipy.stats import kurtosis, skew
from utils_v7 import *

global root_dir, version

# 设置你想要遍历的文件夹路径
root_dir = '../../autodl-tmp/www2024/data_tmp/test'
version = 'v13_2'



def worker_fn(wid,data_lst_wids,):
        
    file_lst = data_lst_wids[wid]
    
    test_df = pd.DataFrame()
    for idx, file in enumerate(file_lst) :
        df_ = pd.read_pickle('{}/{}'.format(root_dir, file))
#         df_ = df_.groupby('file').tail(1000)
        
        df_['t'] = df_['t'].dt.total_seconds()  # 将时间差转换为秒
        df_['a'] = df_['v'] - df_.groupby('file')['v'].shift(1)
        df_['dt'] = df_['t'] - df_.groupby('file')['t'].shift(1)
        
#         df_['dv/dt'] = df_['v'] / df_['dt']
        
        n_init, n_dropna = df_.shape, df_.dropna().shape
        df_nona = df_.dropna().reset_index(drop=True)
        
#         df_ = pd.DataFrame()
        t_cut_lst = [0]
        for idx_t, t_cut in enumerate(t_cut_lst):
            df_ = df_nona.copy()
            df_['dt_rk'] = df_['t'].apply(lambda x: np.floor( (x-t_cut*86400) / (86400*1)) )
            df_ = df_[df_['dt_rk']>=0].reset_index(drop=True)
            df_['t_cut'] = t_cut
#             df_ = pd.concat([df_, df_cut], axis=0)

            df_ = df_.reset_index(drop=True)


            n_filter = df_.shape
            print(n_init, n_dropna, n_filter)

            # 要聚合的字段
            grp_cols = ['file','dt_rk','t_cut']
            
            fields = ['v', 'a', 'dt']
            stats = make_fea(df_, fields, grp_cols)
            
            # 要聚合的字段
            fields = ['t']
            stats_2 = make_fea_simple(df_, fields, grp_cols)
            stats = stats.merge(stats_2, on=grp_cols, how='left')
            
            
            # 要聚合的字段
            df_['v_pos'] = df_['v'] 
            df_['v_neg'] = df_['v'] 
            
            fields = ['v_pos']
            stats_2 = make_fea(df_[df_['v']>=0], fields, grp_cols)
            stats = stats.merge(stats_2, on=grp_cols, how='left')
            
            fields = ['v_neg']
            stats_2 = make_fea(df_[df_['v']<0], fields, grp_cols)
            stats = stats.merge(stats_2, on=grp_cols, how='left')            
            
            test_df = pd.concat([test_df, stats], axis=0)

            del stats,df_,stats_2
            [gc.collect() for _ in range(5)]
            print('===== finished wid-{} part {}/{} t-cut {} / {} ====='.format(wid,idx,len(file_lst), idx_t, len(t_cut_lst) ) )

    print(wid, test_df.shape)
    print(test_df.info())
    test_df.to_pickle('../data_fea_sub/{}/test_X_{}.pkl'.format(version, wid))


if __name__ == '__main__':
    

    file_lst = [filename for filename in os.listdir(root_dir) if filename.endswith('.pkl')][:]
    print(len(file_lst))

    if len(file_lst)>6:
        num_worker = 6

    else:
        num_worker = len(file_lst)

    print('num_worker:', num_worker) 
    
    data_lst_wids = chunk_list(file_lst, len(file_lst) // num_worker, num_worker)
    print('文件切分为：{}块  '.format(len(data_lst_wids)), [len(f) for f in data_lst_wids])

    process_list = []
    for wid in range(num_worker):
        process = mp.Process(target=worker_fn, args=(wid, data_lst_wids,))
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()
  