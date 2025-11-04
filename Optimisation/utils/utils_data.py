# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 21:01:16 2025

@author: Sujingyan

"""
import numpy as np
import re
import os
from scipy.interpolate import NearestNDInterpolator
from copy import deepcopy
import shutil

def mesh_material(file_path,job_name, n_elements = 0):
    
    oname=job_name+'.inp'
    inp_path=os.path.join(file_path,oname)
    nset_dict = {} 
    current_nset_name = None
    current_nset_nodes = []
    reading_nodes = False
    nset_pattern = re.compile(r'^\*Nset.*?nset\s*=\s*([^,\s]+)', re.IGNORECASE)  
    solid_section_pattern = re.compile(r'^\*Solid Section.*elset=([^,\s]+).*material=([^,\s]+)',
                                       re.IGNORECASE) 
    elset_pattern = re.compile(r'^\*Elset.*?elset\s*=\s*([^,\s]+)', re.IGNORECASE)
    node_pattern  = re.compile(r'(\d+)')  
    with open(inp_path, 'r',encoding='gbk') as f:
        
        lines = f.readlines()  
        total_lines = len(lines)
        i = 0
        while i < total_lines:
            line = lines[i].strip()
            i += 1  
            if not line:
                continue      
            if 'ASSEMBLY' in line:
                break
            elset_match = elset_pattern.match(line)
            if elset_match:
                current_nset_name = elset_match.group(1)  
                current_nset_nodes = []
                reading_nodes = True
                continue
            
            nset_match = nset_pattern.match(line)  
            if nset_match and reading_nodes:
                reading_nodes = False
                continue

            if reading_nodes and current_nset_name and not line.startswith('*'):
                nodes = []
                while i <= total_lines and not lines[i - 1].startswith('*'):
                    current_line = lines[i - 1].strip()
                    if current_line:  
                        numbers=node_pattern.findall(current_line)
                        if len(numbers)==3:
                            start = int(numbers[0])
                            end = int(numbers[1])
                            nodes.extend(map(str, range(start, end + 1)))  
                        else:
                            nodes.extend(numbers)
                    i += 1  
                i -= 1  
                if nodes:
                    current_nset_nodes.extend(nodes)
            if current_nset_name and current_nset_nodes:
                if current_nset_name not in nset_dict:  
                    nset_dict[current_nset_name] = current_nset_nodes
        voxel_vectors = []
        new_nset_dict = {}
        all_nset_dict = {}
        i = 0
        elem_sum = 0
        elem_out = 0
        while i < total_lines:
            line = lines[i].strip()
            i += 1  
            if not line:
                continue
            section_match = re.match(r'\*\* Section: (Section-\d+)', line) 
            if section_match:
                current_section = section_match.group(1)  
                continue
            solid_match = solid_section_pattern.match(line)  
            if solid_match:
                elset_name = solid_match.group(1)  
                if elset_name in nset_dict:
                    all_nset_dict[elset_name] = nset_dict[elset_name]
                    if current_section == 'Section-1':
                        voxel_vectors.append(1)
                    elif current_section == 'Section-2':
                        voxel_vectors.append(0)
                    if 'Passive' in elset_name or 'Active' in elset_name:
                        elem_out = elem_out + len( nset_dict[elset_name] )
                    else:
                        new_nset_dict[elset_name] = nset_dict[elset_name] 
                        elem_sum = elem_sum + len( nset_dict[elset_name] )
                        reading_nodes = False
                continue
        
    print('nset_dict:', len(nset_dict), 'all_nset_dict:', len(all_nset_dict))
    print('voxel_nset_dict:', len(new_nset_dict))    
    print('num_elements:', elem_sum, elem_out)    
    
    if n_elements != elem_sum + elem_out:
        raise ValueError("n_elements != elem_sum + elem_out")
    if len(all_nset_dict) == len(voxel_vectors):
        voxel_vectors =np.array(voxel_vectors)
    else:
        raise ValueError("voxel_vectors ERROR!")
        
    return all_nset_dict, new_nset_dict, voxel_vectors


def generate_inputdata(mater_voxels, nset_dict, mesh_nodes, mesh_center_nodes, elem_values = None):
    
    if mater_voxels.shape[1] != len(nset_dict):
        raise ValueError("Please check 'len(mater_voxels)' !")
    # Material voxels -> mesh elements
    if elem_values is not None:
        elem_values = np.repeat(elem_values[np.newaxis, :], mater_voxels.shape[0], axis=0) 
    else:
        elem_values = np.ones((mater_voxels.shape[0], mesh_center_nodes.shape[0]), dtype=int) * -1
    for j in range(mater_voxels.shape[0]):
        for i, (key, value) in enumerate(nset_dict.items()):
            # print(f"{key}: {value}")
            # print(f"{key}")
            elem_index = np.array(value, dtype=int) - 1
            if mater_voxels[j,i] == 1:
                elem_values[j, elem_index] = 1
            elif mater_voxels[j,i] == 0:
                elem_values[j, elem_index] = 0 
            else:
                raise ValueError("Please check 'mater_voxels' !")
    if np.any(elem_values == -1):
        raise ValueError("Matirx 'elem_values' error!")
    
    # Interpolation
    Node_label = np.zeros((elem_values.shape[0], mesh_nodes.shape[0]))
    for i in range(elem_values.shape[0]):

        values = elem_values[i]
        interp_3d = NearestNDInterpolator(mesh_center_nodes, values)
        Node_label[i] = interp_3d(mesh_nodes)
    
    # Cat point coordinate
    if len(Node_label.shape)==2:
        x_data = Node_label.reshape(Node_label.shape[0],-1,1)
        
    Points_expanded = np.tile(mesh_nodes, (x_data.shape[0], 1, 1))
    x_data = np.concatenate([Points_expanded, x_data], axis=-1)
        
    return x_data, elem_values


def inp_file(save_path, base_inp_path, mater_voxels, nset_dict,
             modify_section=True, modify_material=True,output_extension='.txt'):
    
    keys = list(nset_dict.keys())
    if mater_voxels.shape[1] != len(nset_dict):
        raise ValueError("Please check 'len(mater_voxels)' !")
    current_section = None  # "Section-1"/"Section-2"
    new_section=None
    modified_lines=[]
    base_name = os.path.splitext(os.path.basename(base_inp_path))[0]#curvedsurface
    output_path = os.path.join(save_path, base_name+output_extension)
    print(output_path)
    solid_section_pattern = re.compile(r'^\*Solid Section.*elset=([^,\s]+).*material=([^,\s]+)',
                                       re.IGNORECASE)  
    All_sections = []
    with open(base_inp_path, 'r', encoding='gbk') as f:
        lines = f.readlines()
        modified_lines = deepcopy(lines)
        total_lines = len(lines)
        i = 0
        n_voxel = 0
        while i < total_lines:
            
            line = modified_lines[i].strip()
            i += 1 

            if not line:
                continue

            section_match = re.match(r'\*\* Section: (Section-\d+)', line) 
            if section_match and modify_section:
                solid_match = solid_section_pattern.match(modified_lines[i].strip()) 
                elset_name  = solid_match.group(1)
                if 'Passive' in elset_name or 'Active' in elset_name:
                    continue
                
                current_section = section_match.group(1)
                All_sections.append(current_section)
                if mater_voxels[0, n_voxel] == 1:
                    new_section = 'Section-1'
                elif mater_voxels[0, n_voxel] == 0:
                    new_section = 'Section-2'
                else:
                    raise ValueError("mater_voxels error!")
                key_name = keys[n_voxel]
                n_voxel = n_voxel + 1
                modified_lines[i-1] =  modified_lines[i-1].replace(current_section, new_section)
    
                continue
            
            solid_match = solid_section_pattern.match(line)  
            if solid_match and modify_material:
                elset_name = solid_match.group(1)  
                material_name = solid_match.group(2) 
                if 'Passive' in elset_name or 'Active' in elset_name:
                    continue
                if elset_name != key_name:
                    print(elset_name, key_name)
                    raise ValueError("(elset_name, key_name) error!")
                if new_section=='Section-1':
                    new_material='Material-active'
                    modified_lines[i-1] = modified_lines[i-1].replace(material_name, new_material)
                elif new_section=='Section-2':
                    new_material='Material-passive'
                    modified_lines[i-1] = modified_lines[i-1].replace(material_name, new_material)
                    
                continue
            
    if len(All_sections) != len(keys):
        raise ValueError("len(All_sections) != len(keys)！")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='gbk') as f_out:
        f_out.writelines(modified_lines)
    print('the new inp path:', output_path)
    