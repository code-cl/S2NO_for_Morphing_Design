# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 20:47:45 2025

@author: Sujingyan

"""

import scipy.io as sio
import os
import warnings
import numpy as np
import random
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils.utils_predict import load_model, Predicting
from utils.utils_data import mesh_material, generate_inputdata, inp_file
from utils.deap import base, creator, tools, algorithms

def evaluate(individual, target_shape, nset_dict, mesh_nodes, mesh_center_nodes, 
             model, norm_x, norm_y, y_dim, face_ids, intial_elem_values):
    
    mater_voxels = np.array(individual).reshape(1,-1)
    x_data,_   = generate_inputdata(mater_voxels, nset_dict, mesh_nodes, mesh_center_nodes, elem_values = intial_elem_values[0])
    
    pre_test = Predicting(x_data, model, norm_x, norm_y, y_dim)
    pre_data = pre_test[0, face_ids,:]
    target_data = target_shape[0]

    #value    = np.max(np.linalg.norm(pre_data - target_data, ord=2, axis=1), axis = 0)
    value    = np.mean(np.linalg.norm(pre_data - target_data, ord=2, axis=1), axis = 0)
    
    return (value,)

    

if __name__ == "__main__":
    
    case = 'Blade'
    file_path  = './model/'+case+'/'
    save_path  =  file_path + 'S2NO_model/' 
    lbo_path  = file_path +'Nodes_LBO_basis'
    result_folder = './results/'+case+'/'
    
    if not os.path.exists(save_path + 'model_params.pkl'):
        raise ValueError("No File:", save_path + 'model_params.pkl')
    lbo_data = sio.loadmat(lbo_path)
    nodes  = lbo_data['Points']
    elems  = lbo_data['Elements']
    elems  = np.hstack((np.full((elems.shape[0], 1), elems.shape[1]), elems))
    print('nodes:', nodes.shape, elems.shape, np.min(elems))
    print('elems:', elems.shape, np.min(elems))
    
    args = dict()
    args['model_type']   = 'S2NO_spec' # S2NO_spec, S2NO
    args['n_layers']     = 8
    args['num_lbos']     = 128
    args['num_channels'] = 128
    args['num_heads']    = 8
    args['mlp_ratio']    = 1
    args['lbo_path']  = lbo_path
    args['norm_type'] = 'coeff_norm'  # 'coeff_norm' or 'point_norm' or 'no_norm
    args['space_dim'] = 3
    args['x_dim'] = 4
    args['y_dim'] = 3
    args['nodes']  = nodes #None
    args['elems']  = elems #None
    args['save_path'] = save_path
    args['placeholder']   = True #True
    args['initialize_weights'] = True
    args['device'] = "cuda"
    args['step_size'] = 100
    args['gamma']        = 0.5
    args['xlbo_path']  = lbo_path
    args['ylbo_path']  = lbo_path
    
    mesh_data = sio.loadmat(file_path + '/node_mesh.mat')
    mesh_nodes  = mesh_data["nodes"][:, 1:]
    mesh_center_nodes = mesh_data['mesh_center'][:, 1:]
    
    model, norm_x, norm_y = load_model(args)
    all_nset, nset_dict, intial_mater = mesh_material(file_path,'curvedsurface', mesh_center_nodes.shape[0])
    _, intial_elem_values  = generate_inputdata(intial_mater.reshape(1,-1), all_nset, mesh_nodes, mesh_center_nodes)
    

    target_names = ['F_target_7']
    for target_name in target_names:
        
        result_path = result_folder + target_name
        print('\n' + result_path)
        target_data = sio.loadmat(file_path + target_name + '.mat')
        face_ids     = target_data['face_ids'][0]
        target_shape = target_data['face_nodes']
        target_shape = target_shape[np.newaxis,...]
        
       
        if 1:
           
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
            toolbox = base.Toolbox()

            NDIM = len(nset_dict)
        
            toolbox.register("attr_bool", random.randint, 0, 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, NDIM)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            toolbox.register("evaluate", evaluate, target_shape = target_shape, nset_dict = nset_dict, 
                                                    mesh_nodes = mesh_nodes, mesh_center_nodes = mesh_center_nodes, model = model, 
                                                    norm_x = norm_x, norm_y = norm_y, y_dim = args['y_dim'], face_ids = face_ids, 
                                                    intial_elem_values = intial_elem_values
                                                    )
        
            
            toolbox.register("mate", tools.cxTwoPoint)                
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)   
            toolbox.register("select", tools.selTournament, tournsize=2)
            
            pop = toolbox.population(n=1000)      
            hof = tools.HallOfFame(1)    
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
        
            pop, log = algorithms.eaSimple(
                                            pop, 
                                            toolbox, 
                                            cxpb=0.75, 
                                            mutpb=0.2, 
                                            ngen=100, 
                                            stats=stats, 
                                            halloffame=hof, 
                                            verbose=True
                                          )
        
            best_ind = hof[0]
            print(f"\nOptimal solution: {best_ind}")
            print(f"Optimal value: {best_ind.fitness.values[0]}")
                
            x_, e_   = generate_inputdata(np.array(best_ind).reshape(1,-1), nset_dict, 
                                          mesh_nodes, mesh_center_nodes, elem_values = intial_elem_values[0])
            pre_ = Predicting(x_, model, norm_x, norm_y, args['y_dim'])
            
            inp_save_path = result_path
            base_inp_path = file_path + 'curvedsurface.inp'
            inp_file(inp_save_path, base_inp_path, np.array(best_ind).reshape(1,-1), nset_dict)
        
            new_txt_file = inp_save_path + '/curvedsurface.txt'
            inp_path     = os.path.splitext(new_txt_file)[0] + ".inp"  
            os.rename(new_txt_file, inp_path)
            
            sio.savemat(result_path + '/results_ea.mat', {  'best_ind'  : best_ind,
                                                            'opt_voxels': np.array(best_ind).reshape(1,-1),
                                                            'x_opt'     : x_,
                                                            'e_opt'     : e_,
                                                            'y_opt'     : pre_,
                                                            'target' : target_shape
                                                        })
        