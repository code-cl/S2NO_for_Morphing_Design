# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:09:22 2025

@author: Chenl
"""
import torch
import numpy as np
import scipy.io as sio
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    
    data_file = 'def_data'
    file_path = '../data/Blade/' 
    save_path = 'logs/Blade'
    lbo_path  = file_path +'Nodes_LBO_basis'
    # Read data
    lbo_data = sio.loadmat(lbo_path)
    nodes  = lbo_data['Points']
    elems  = lbo_data['Elements']
    elems  = np.hstack((np.full((elems.shape[0], 1), elems.shape[1]), elems))
    print('nodes:', nodes.shape, elems.shape, np.min(elems))
    print('elems:', elems.shape, np.min(elems))
    
    n_floders = 2
    x_datas = []
    y_datas = []
    n_train = 0
    for i in range (n_floders):
        print(file_path + str(n_floders-i-1)+ '/' + data_file +'.mat')
        x_da = sio.loadmat(file_path + str(n_floders-i-1) + '/mater_data.mat')     ["mater"]
        y_da = sio.loadmat(file_path + str(n_floders-i-1) + '/' + data_file +'.mat')["defor"]
        
        if n_floders-i-1 != 0:
            print(x_da.shape)
            n_train = x_da.shape[0] + n_train
        else:
            print('all_test_data:', x_da.shape)
            print('n_train:', n_train)
        x_datas.append(x_da)
        y_datas.append(y_da)
        
    x_data = np.concatenate(x_datas, axis=0)
    y_data = np.concatenate(y_datas, axis=0)
    print(x_data.shape, y_data.shape)
    
    if len(x_data.shape)==2:
        x_data = x_data.reshape(x_data.shape[0],-1,1)
    Points_expanded = np.tile(nodes, (x_data.shape[0], 1, 1))
    x_data = np.concatenate([Points_expanded, x_data], axis=-1)
    y_data = y_data + nodes
    
    # Set model parameter
    args = dict()
    args['model_type']   = 'S2NO_spec' # S2NO, S2NO_spec
    args['n_layers']     = 8
    args['num_lbos']     = 128
    args['num_channels'] = 128
    args['num_heads']    = 8
    args['mlp_ratio']    = 1
    args['lbo_path']  = lbo_path
    args['norm_type'] = 'coeff_norm'  # 'coeff_norm' or 'point_norm' or 'no_norm
    args['space_dim'] = nodes.shape[-1]
    args['x_dim'] = x_data.shape[-1]
    if len(y_data.shape)==2:
        args['y_dim'] = 1
    elif len(y_data.shape)==3:
        args['y_dim'] = y_data.shape[-1]
    
    # Set data
    args['xdata']  = x_data
    args['ydata']  = y_data
    args['nodes']  = nodes # None
    args['elems']  = elems # None
    args['n_train'] = (n_floders - 1)*10000
    args['n_test' ] = 5000
    args['save_tedata_size'] = args['n_test' ]
    args['save_trdata_size'] = 100
    
    # Set training parameter
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['loss']   = 'L2' # L2 or SSE
    args['batch_size']   = 16
    args['learn_rate']   = 0.001
    args['epoch'  ]      = 500
    args['max_grad_norm'] = 0.1
    args['placeholder']   = True 
    args['initialize_weights'] = True
    args['optimizer'] = 'AdamW' 
    args['scheduler'] = 'OneCycleLR' 
    args['weight_decay'] = 1e-5
    args['save_path'] = save_path 
    

    from utils.utils_run_OneCycleLR import Dataset, Train, Test
        
    data_array = Dataset(args)
    x_train, y_train, x_test, y_test = data_array[0], data_array[1], data_array[2], data_array[3]

    txt_path = args['save_path'] + "/args.txt"
    with open(txt_path, "w") as f:
        # f.write(f"{'n_floders'}: {n_floders}\n")
        for key, value in args.items():
            if key == 'xdata' or key == 'ydata' or key == 'nodes' or key == 'elems' or key == 'Data_modes':
                continue  # 跳过 'data'
            f.write(f"{key}: {value}\n")
    
    model = Train(args, data_array)
    Test(args, data_array, model)