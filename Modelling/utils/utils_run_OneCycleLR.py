
import os
import torch
import numpy as np
import scipy.io as sio
import time
from utils.utilities3 import LpLoss,UnitGaussianNormalizer, GaussianNormalizer, WithOutNormalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
        # print(f"{name:<40} | {str(param.size()):<20} | {param_count:<10}")
    # print(f"Total Trainable Params: {total_params}")
    return total_params

def Dataset(args):
    
    print('\n*************** Get Dataset ***************')
    x_data = args['xdata']
    y_data = args['ydata']
    device = args['device']
    
    if args['n_train']+args['n_test']>x_data.shape[0]:
        raise ValueError("Please check 'num of data' !", args['n_train']+args['n_test'], x_data.shape[0])
        
    x_train = torch.Tensor(x_data[:args['n_train']]).to(device)
    y_train = torch.Tensor(y_data[:args['n_train']]).to(device)
    x_test  = torch.Tensor(x_data[-args['n_test']:]).to(device)
    y_test  = torch.Tensor(y_data[-args['n_test']:]).to(device)
    del x_data, y_data
    gc.collect()
    
    if len(x_train.shape)!=3:
        raise ValueError("Please check 'dim of X' !")

    if args['norm_type'] == 'coeff_norm':
        norm_x  = GaussianNormalizer(x_train)
        x_train = norm_x.encode(x_train)
        x_test  = norm_x.encode(x_test)
        
        norm_y  = GaussianNormalizer(y_train)
        y_train = norm_y.encode(y_train)
        y_test  = norm_y.encode(y_test)
    
    elif args['norm_type'] == 'point_norm':
        
        norm_x  = UnitGaussianNormalizer(x_train)
        x_train = norm_x.encode(x_train)
        x_test  = norm_x.encode(x_test)
        
        norm_y  = UnitGaussianNormalizer(y_train)
        y_train = norm_y.encode(y_train)
        y_test  = norm_y.encode(y_test)
    
    elif args['norm_type'] == 'no_norm':
        
        norm_x  = WithOutNormalizer(x_train)
        x_train = norm_x.encode(x_train)
        x_test  = norm_x.encode(x_test)
        
        norm_y  = WithOutNormalizer(y_train)
        y_train = norm_y.encode(y_train)
        y_test  = norm_y.encode(y_test)
    
    else:
        raise ValueError("Please check 'norm_type' !")
    
    print('norm_x:', norm_x.mean, norm_x.std)
    print('norm_y:', norm_y.mean, norm_y.std)
    
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
        
    torch.save({'mean': norm_x.mean, 'std': norm_x.std, 'eps': norm_x.eps}, args['save_path'] + "/" + 'norm_x.pth')
    torch.save({'mean': norm_y.mean, 'std': norm_y.std, 'eps': norm_y.eps}, args['save_path'] + "/" + 'norm_y.pth')
    
    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    print('x_test:' , x_test.shape , 'y_test:' , y_test.shape)
    
    return x_train, y_train, x_test, y_test, norm_x, norm_y
    

def Train(args, data_array):
    
    print('\n*************** Training ***************')
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
        
    x_train, y_train, x_test, y_test = data_array[0], data_array[1], data_array[2], data_array[3]
    _, norm_y = data_array[4],data_array[5]
    device_norm = norm_y.mean.device
    
    n_train    = args['n_train']
    n_test     = args['n_test']
    device     = args['device']
    print(device, device_norm)
    
    if 'S2NO' in args['model_type']:
        from   utils.S2NO import Model
        model = Model(args).to(device)
        file_name = 'S2NO'
    else:
        raise ValueError("Please check 'model_type' !")    
    
    print('Num of paras : %d'%(count_parameters(model)))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                                batch_size=args['batch_size'], shuffle=True)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                                batch_size=args['batch_size'], shuffle=False)
    
    del x_train, y_train, x_test, y_test
    gc.collect()
    
    if args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learn_rate'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args['learn_rate'], weight_decay=args['weight_decay'])
    else:
        raise ValueError("Please check 'optimizer' !")   
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args['learn_rate'], epochs=args['epoch'],
                                                    steps_per_epoch=len(train_loader))
    
    myloss = LpLoss(size_average=False)
    train_error = np.zeros((args['epoch']))
    test_error  = np.zeros((args['epoch']))

    for ep in range(args['epoch']):
        model.train()
        time_step = time.perf_counter()
        train_l2 = 0
        tr_maxe = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            
            if args['y_dim'] == 1:
                out = out.view(out.shape[0], -1)
            out_real = norm_y.decode(out.to(device_norm))
            y_real   = norm_y.decode(y.to(device_norm))
            
            if args['loss'] == 'L2':
                l2 = myloss(out, y)  # 逆归一化前的L2
            elif args['loss'] == 'SSE':
                l2 = torch.mean(torch.sum(torch.norm(y_real - out_real, p=2, dim=2), dim=1)) # 逆归一化后的SSE
            else:
                raise ValueError("Please check 'loss_type' !")
            l2.backward() 
            if args['max_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
            optimizer.step()
            train_l2 += myloss(out_real, y_real).item()   
            
            if args['y_dim'] == 1:
                tr_maxe.append(torch.max(y_real - out_real, dim=1).values)
            else:
                tr_maxe.append(torch.max(torch.norm(y_real - out_real, p=2, dim=2), dim=1).values)
            scheduler.step()  # OneCycleLR
        tr_maxe_tensor = torch.cat(tr_maxe).squeeze()
        train_l2 /= n_train
        train_error[ep] = train_l2
        tr_mmax = torch.mean(tr_maxe_tensor).item()
        time_step_1 = time.perf_counter()
        torch.save(model.state_dict(), args['save_path'] + "/" + 'model_params.pkl')
        
        if (ep != 0 and ep%1 == 0) or ep == args['epoch'] - 1:
            
            model.eval()
            test_l2 = 0.0
            te_maxe = []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    
                    if args['y_dim'] == 1:
                        out = out.view(out.shape[0], -1)
                    out_real = norm_y.decode(out.to(device_norm))
                    y_real   = norm_y.decode(y.to(device_norm))
                    
                    test_l2 += myloss(out_real, y_real).item()      
                    if args['y_dim'] == 1:
                        te_maxe.append(torch.max(y_real - out_real, dim=1).values)
                    else:
                        te_maxe.append(torch.max(torch.norm(y_real - out_real, p=2, dim=2), dim=1).values)
                   
            te_maxe_tensor = torch.cat(te_maxe).squeeze()
            test_l2  /= n_test
            test_error[ep]  = test_l2
            te_mmax = torch.mean(te_maxe_tensor).item() 
            time_step_2 = time.perf_counter()
            print('Step: %d, Train L2: %.5f, Test L2: %.5f, Train mmax: %.5f, Test mmax: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, tr_mmax, te_mmax, time_step_2 - time_step))
             
            loss_dict = {'train_error' :train_error,
                         'test_error'  :test_error}
            sio.savemat(args['save_path'] +'/'+file_name+'_loss.mat', mdict = loss_dict)   
            
        else:
            print('Step: %d, Train L2: %.5f, Train mmax: %.5f, Time: %.3fs'%(ep, train_l2, tr_mmax, time_step_1 - time_step))
        # if ep!=0 and ep%100==0:
        #     Test(args, data_array, model, save_type=str(ep))
    print('\nTesting error: %.3e'%(test_error[-1])) # after training
    print('Training error: %.3e'%(train_error[-1])) # after i-1 training
    print('Training END!')
    
    del train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return model


def Test(args, data_array, model, save_type = "normal"):
    
    print('\n*************** Testing ***************')
    if 'S2NO' in args['model_type']:
        file_name = 'S2NO'
    else:
        raise ValueError("Please check 'model_type' !") 
        
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    device = args['device']    
    x_train, y_train, x_test, y_test = data_array[0], data_array[1], data_array[2], data_array[3]
    norm_x, norm_y = data_array[4],data_array[5]
    device_norm = norm_y.mean.device
    shape_list = []
    shape_list.append(x_train.shape)
    shape_list.append(y_train.shape)
    shape_list.append(x_test.shape)
    shape_list.append(y_test.shape)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    
    del x_train, y_train, x_test, y_test
    gc.collect()
    
    pre_train = torch.zeros(shape_list[1])
    y_train   = torch.zeros(shape_list[1])
    x_train   = torch.zeros(shape_list[0])
    
    index = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            
            if args['y_dim'] == 1:
                out = out.view(1, -1)
            out_real = norm_y.decode(out.to(device_norm))
            y_real   = norm_y.decode(y.to(device_norm))
            
            x_train[index] = norm_x.decode(x.to(device_norm))
            if args['y_dim'] == 1:
                pre_train[index] = out_real.reshape(-1)
                y_train[index]   = y_real.reshape(-1)
            else:
                pre_train[index] = out_real
                y_train[index]   = y_real
            index = index + 1       
    
    pre_test = torch.zeros(shape_list[3])
    y_test   = torch.zeros(shape_list[3])
    x_test   = torch.zeros(shape_list[2])
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            
            if args['y_dim'] == 1:
                out = out.view(1, -1)
            out_real = norm_y.decode(out.to(device_norm))
            y_real   = norm_y.decode(y.to(device_norm))
            x_real   = norm_x.decode(x.to(device_norm))
            
            x_test[index] = x_real
            if args['y_dim'] == 1:
                pre_test[index] = out_real.reshape(-1)
                y_test[index]   = y_real.reshape(-1)
            else:
                pre_test[index] = out_real
                y_test[index]   = y_real
                
            index = index + 1

    # ================ Save Data ====================
    myloss = LpLoss(size_average=False)
    test_l2  = (myloss( pre_test,  y_test).item()) / y_test.shape[0]
    train_l2 = (myloss( pre_train, y_train).item())/ y_train.shape[0]
    
    if args['y_dim'] == 1:
        tr_maxes = torch.max(y_train - pre_train, dim=1).values
        te_maxes = torch.max(y_test - pre_test, dim=1).values
    else:
        tr_maxes = torch.max(torch.norm(y_train - pre_train, p=2, dim=2), dim=1).values
        te_maxes = torch.max(torch.norm(y_test - pre_test, p=2, dim=2), dim=1).values
    
    tr_maxe  = torch.max(tr_maxes).item()
    tr_mmaxe = torch.mean(tr_maxes).item()
    
    te_maxe  = torch.max(te_maxes).item()
    te_mmaxe = torch.mean(te_maxes).item()
    
    txt_path =  args['save_path'] + '/log.txt'
    with open(txt_path, "w") as f:
        f.write(f"{'Test_loss'} : {test_l2}\n")
        f.write(f"{'Test_MAE'}  : {mean_absolute_error(y_test.ravel().cpu().detach().numpy(), pre_test.ravel().cpu().detach().numpy())}\n")
        f.write(f"{'Test_MMax'} : {te_mmaxe}\n")
        f.write(f"{'Test_Max'}  : {te_maxe}\n")
        f.write("\n")
        f.write(f"{'Train_loss'}: {train_l2}\n")
        f.write(f"{'Train_MAE'}  : {mean_absolute_error(y_train.ravel().cpu().detach().numpy(), pre_train.ravel().cpu().detach().numpy())}\n")
        f.write(f"{'Train_MMax'}: {tr_mmaxe}\n")
        f.write(f"{'Train_Max'} : {tr_maxe}\n")
        f.write(f"{'num_paras'} : {count_parameters(model)}\n")

    pred_dict = {
                    'pre_test'  : pre_test[0:args['save_tedata_size']].cpu().detach().numpy(),
                    'x_test'    : x_test[0:args['save_tedata_size']].cpu().detach().numpy(),
                    'y_test'    : y_test[0:args['save_tedata_size']].cpu().detach().numpy(),
                    'x_train'   : x_train [0:args['save_trdata_size']].cpu().detach().numpy(),
                    'y_train'   : y_train [0:args['save_trdata_size']].cpu().detach().numpy(),
                    'pre_train' : pre_train[0:args['save_trdata_size']].cpu().detach().numpy()
                }
                                             
    if save_type == "normal" :                                        
        sio.savemat(args['save_path'] + '/' + file_name + '_result.mat' , mdict = pred_dict)
    else:
        sio.savemat(args['save_path'] +'/' + file_name + '_result_'+save_type +'.mat' , mdict = pred_dict)
    
    print('\nTesting error: %.3e'%(test_l2))
    print('Testing MAError: %.3e'%(mean_absolute_error(y_test.ravel().cpu().detach().numpy(), pre_test.ravel().cpu().detach().numpy())))
    print('Testing MeanMax: %.3e'%(te_mmaxe))
    print('Testing MaxError: %.3e'%(te_maxe))
    
    print('\nTraining error: %.3e'%(train_l2))
    print('Training MAError: %.3e'%(mean_absolute_error(y_train.ravel().cpu().detach().numpy(), pre_train.ravel().cpu().detach().numpy())))
    print('Training MeanMax: %.3e'%(tr_mmaxe))
    print('Training MaxError: %.3e'%(tr_maxe))
    print('Num of paras : %d'%(count_parameters(model)))
    
    del x_train, y_train, x_test, y_test, pre_train, pre_test, train_loader, test_loader  
    gc.collect()       
    torch.cuda.empty_cache()  