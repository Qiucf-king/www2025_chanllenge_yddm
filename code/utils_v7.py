import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import multiprocessing as mp
import gc
from scipy.stats import kurtosis, skew

def chunk_list(input_list, chunk_size, num_worker):
    total_elements = len(input_list)
    
    # 如果元素总数小于工作线程数，每个线程处理一个元素
    if total_elements < num_worker:
        chunk_size = 1
        num_worker = total_elements  # 更新工作线程数为元素总数

    # 计算每个工作线程应处理的元素数量
    ideal_chunk_size = total_elements // num_worker
    remainder = total_elements % num_worker
    
    new_lst = []
    start_index = 0
    
    for i in range(num_worker):
        # 每个线程分配的块大小
        current_chunk_size = ideal_chunk_size + (1 if i < remainder else 0)
        end_index = start_index + current_chunk_size
        
        # 确保不超出列表范围
        new_lst.append(input_list[start_index:end_index])
        
        start_index = end_index  # 更新起始索引
    print(total_elements, sum( [len(f) for f in new_lst] ), [len(f) for f in new_lst])
    return new_lst
    
def make_fea(df_slc, fields, grp_cols):
    # 创建聚合操作的字典
    agg_operations = {}
    for field in fields:
        agg_operations[f'{field}_mean'] = (field, 'mean')
        agg_operations[f'{field}_std'] = (field, 'std')
        agg_operations[f'{field}_max'] = (field, 'max')
        agg_operations[f'{field}_min'] = (field, 'min')
        agg_operations[f'{field}_p2p'] = (field, lambda x: x.max() - x.min())
        agg_operations[f'{field}_unicnt'] = (field, lambda x: x.nunique())
        agg_operations[f'{field}_count'] = (field, 'count')
        
        # 添加四分位数
        agg_operations[f'{field}_Q25'] = (field, lambda x: x.quantile(0.25))
        agg_operations[f'{field}_Q50'] = (field, lambda x: x.quantile(0.50))  # 中位数
        agg_operations[f'{field}_Q75'] = (field, lambda x: x.quantile(0.75))
        
        # 添加偏度和峰度
        agg_operations[f'{field}_skew'] = (field, lambda x: skew(x.dropna()))
        agg_operations[f'{field}_kurt'] = (field, lambda x: kurtosis(x.dropna(), fisher=True))  # fisher=True 返回相对于正态分布的峰度
        
#         # 添加FFT特征
#         agg_operations[f'{field}_fft_max'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].max())
#         agg_operations[f'{field}_fft_2nd_max'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].argsort()[-2])
#         agg_operations[f'{field}_fft_3rd_max'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].argsort()[-3])
        
#         agg_operations[f'{field}_fft_mean'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].mean())
#         agg_operations[f'{field}_fft_std'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].std())
#         agg_operations[f'{field}_fft_4rd_max'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].argsort()[-4])
#         agg_operations[f'{field}_fft_5rd_max'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].argsort()[-5])
        
#         agg_operations[f'{field}_fft_max_diff_1_2'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].max() - np.abs(np.fft.fft(x.dropna()))[:len(x)//2].argsort()[-2])
#         agg_operations[f'{field}_fft_max_diff_2_3'] = (field, lambda x: np.abs(np.fft.fft(x.dropna()))[:len(x)//2].argsort()[-2] - np.abs(np.fft.fft(x.dropna()))[:len(x)//2].argsort()[-3])
        

    # 根据 'file' 字段分组并计算统计值
    stats = df_slc.groupby(grp_cols).agg(**agg_operations).reset_index()
    
    return stats



def make_fea_simple(df_slc, fields, grp_cols):
    # 创建聚合操作的字典
    agg_operations = {}
    for field in fields:

        agg_operations[f'{field}_max'] = (field, 'max')
        agg_operations[f'{field}_min'] = (field, 'min')
        agg_operations[f'{field}_p2p'] = (field, lambda x: x.max() - x.min())

    # 根据 'file' 字段分组并计算统计值
    stats = df_slc.groupby(grp_cols).agg(**agg_operations).reset_index()
    
    return stats