import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import multiprocessing as mp



global root_dir

# 设置你想要遍历的文件夹路径
root_dir = '../data/train_X'

def read_pkl(file_path):
    """
    读取.pkl文件并返回数据。
    
    参数:
    file_path (str): .pkl文件的路径。
    
    返回:
    object: 从.pkl文件中加载的数据。
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def write_pkl(data, file_path):
    """
    将数据写入.pkl文件。
    
    参数:
    data (object): 要写入的数据。
    file_path (str): .pkl文件的路径。
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
        
def make_df(folder_path, file):
    # file = file_lst[130]
    file_all = os.path.join(folder_path, file)
    data = read_pkl(file_all)
    df_ = pd.DataFrame()
    df_['t'] = data['t']
    df_['v'] = data['v']
    df_.insert(0, 'file', file)
    return df_




def chunk_list(input_list, chunk_size, num_worker):
    new_lst = [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    if len(new_lst)>num_worker:
        print(1)
        new_lst_2 = new_lst[:num_worker].copy()
        print([len(f) for f in new_lst_2])
        print([len(f) for f in new_lst])
        new_lst_2[-1] = [f for f in new_lst_2[-1]] + [f for f in new_lst[-1]]
        return new_lst_2
    else:
        return new_lst


    
def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def worker_fn(wid,data_lst_wids,):
        
    file_lst = data_lst_wids[wid]
    
    df = pd.DataFrame()
    for idx, file in tqdm(enumerate(file_lst) ):
        df_ = make_df(root_dir, file)
        df = pd.concat([df, df_], axis=0)

        print('===== finished wid-{} part {}/{} ====='.format(wid,idx,len(file_lst) ) )
    df.to_pickle('../data_tmp/train/part_{}.pkl'.format(wid) )
    print(wid, df.shape)

if __name__ == '__main__':
    


    file_lst = [filename for filename in os.listdir(root_dir) if filename.endswith('.pkl')][:]
    print(len(file_lst))

    if len(file_lst)>12:
        num_worker = 12

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
  