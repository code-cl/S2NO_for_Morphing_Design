# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:09:22 2025

@author: Chenl
"""
import torch
from   utils.utilities3 import count_params, LpLoss, UnitGaussianNormalizer, GaussianNormalizer, WithOutNormalizer
from sklearn.metrics import mean_absolute_error, r2_score

def load_norm(args):
    
    norm_x_ = torch.load(args['save_path'] + "/" + 'norm_x.pth')
    norm_y_ = torch.load(args['save_path'] + "/" + 'norm_y.pth')
    
    dummy_x = torch.zeros(1, *norm_x_['mean'].shape)
    dummy_y = torch.zeros(1, *norm_y_['mean'].shape)
    
    if args['norm_type'] == 'coeff_norm':
        norm_x  = GaussianNormalizer(dummy_x, eps=norm_x_['eps'])
        norm_y  = GaussianNormalizer(dummy_y, eps=norm_y_['eps'])
    
    elif args['norm_type'] == 'point_norm':
        norm_x  = UnitGaussianNormalizer(dummy_x, eps=norm_x_['eps'])
        norm_y  = UnitGaussianNormalizer(dummy_y, eps=norm_y_['eps'])
    
    elif args['norm_type'] == 'no_norm':
        norm_x  = WithOutNormalizer(dummy_x, eps=norm_x_['eps'])
        norm_y  = WithOutNormalizer(dummy_y, eps=norm_y_['eps'])
    
    else:
        raise ValueError("Please check 'norm_type' !")
   
    norm_x.mean = norm_x_['mean'].cuda()
    norm_x.std  = norm_x_['std'].cuda()
    
    norm_y.mean = norm_y_['mean'].cuda()
    norm_y.std  = norm_y_['std'].cuda()
    
    print('norm_x:', norm_x.mean, norm_x.std)
    print('norm_y:', norm_y.mean, norm_y.std)
    
    # x_test  = norm_x.encode(x_test)
    # y_test  = norm_y.encode(y_test)
    
    # print('x_test:' , x_test.shape , 'y_test:' , y_test.shape)
    
    return norm_x, norm_y


def load_model(args):
    
    print('\n========== Load model ==========')
    
    norm_x, norm_y = load_norm(args)
    current_device = torch.cuda.current_device()
    
    if 'S2NO' in args['model_type']:
        from   utils.S2NO import Model
        model = Model(args).cuda()
    else:
        raise ValueError("Please check 'model_type' !")  
    print('Num of paras : %d'%(count_params(model)))
    model.load_state_dict(torch.load(args['save_path'] + "/" + 'model_params.pkl', map_location=f"cuda:{current_device}"))  
    model.eval()
    
    return model, norm_x, norm_y
    
def Predicting(x_test, model, norm_x, norm_y, y_dim = 3):
    
    if len(x_test.shape)==2:
        x_test  = x_test.reshape(x_test.shape[0],-1,1)
        
    elif len(x_test.shape)!=3:
        raise ValueError("Please check 'dim of X' !")
    
    # print(norm_x.mean.device)
    x_test = norm_x.encode(torch.Tensor(x_test).cuda())
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test), 
                                              batch_size=1, shuffle=False)
    
    pre_test = torch.zeros(x_test.shape[0], x_test.shape[1], y_dim)
    
    index = 0
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.cuda()

            out = model(x)
            
            if y_dim == 1:
                out = out.view(1, -1)
                
            out_real = norm_y.decode(out)

            if y_dim  == 1:
                pre_test[index] = out_real.reshape(-1)
            
            else:
                pre_test[index] = out_real
                
            index = index + 1
    
    return pre_test.cpu().numpy()



def Evaluate_error(y_test, pre_test, save_path):
    
    y_test = torch.Tensor(y_test)
    pre_test = torch.Tensor(pre_test)
    
    myloss = LpLoss(size_average=False)
    test_l2  = (myloss( pre_test,  y_test).item()) / y_test.shape[0]
    
    te_maxes = torch.max(torch.norm(y_test - pre_test, p=2, dim=2), dim=1).values
    te_maxe  = torch.max(te_maxes).item()
    te_mmaxe = torch.mean(te_maxes).item()
    sse_error = torch.mean(torch.sum(torch.norm(y_test - pre_test, p=2, dim=2), dim=1)).item()
    print('\nError L2: %.3e'%(test_l2))
    print('MAError: %.3e'%(mean_absolute_error(y_test.ravel(), pre_test.ravel())))
    print('MeanMax: %.3e'%(te_mmaxe))
    print('MaxError: %.3e'%(te_maxe))
    print('R2_score:', f"X:{r2_score(y_test[:,0], pre_test[:,0]):.3f}",
                            f"Y:{r2_score(y_test[:,1], pre_test[:,1]):.3f}",
                            f"Z:{r2_score(y_test[:,2], pre_test[:,2]):.3f}")
    
    print('SumP2P: %.3e'%(sse_error))
    
