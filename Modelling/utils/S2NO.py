
from timm.models.layers import trunc_normal_
import torch
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch_geometric.nn as gnn
from einops import rearrange
from utils.lapy import Solver, TriaMesh, TetMesh

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.placeholder = args['placeholder']
        n_layers  = args['n_layers']
        input_dim = args['x_dim']
        out_dim   = args['y_dim']
        num_channels = args['num_channels']
        num_lbos     = args['num_lbos']
        space_dim    = args['space_dim']
        num_heads    = args['num_heads']
        device       = args['device']
        model_type   = args['model_type']
        print('Model is ', model_type)
        
        lbo_data = sio.loadmat(args['lbo_path'])['Eigenvectors'][:,:num_lbos]
        if num_lbos > lbo_data.shape[1]:
            raise ValueError("Please check 'num of lbo' !")
        lbo_bases = torch.Tensor(lbo_data).to(device)
        lbo_bases = F.normalize(lbo_bases, p = 2, dim = -1 ) # L2 Norm
        lbo_inver = (lbo_bases.T @ lbo_bases).inverse() @ lbo_bases.T
        print('lbo_bases:', lbo_bases.shape, 'lbo_inver:', lbo_inver.shape)
        
        self.fc0 = nn.Linear(input_dim, num_channels) 
        self.fc1 = nn.Linear(num_channels, 128)
        self.fc2 = nn.Linear(128, out_dim)
        
        # self.ln  = nn.LayerNorm(num_channels)
        # self.mlp = nn.Linear(num_channels, out_dim)
        
        if args['nodes'] is not None :
            nodes = args['nodes']
            elems = args['elems']

            if nodes.shape[-1] == 2:
                nodes = np.hstack((nodes, np.full((nodes.shape[0], 1), 0)))
            if elems.shape[-1] == 4:
                mesh = TriaMesh(nodes, elems[:,1:])
                fem  = Solver(mesh)
            elif elems.shape[-1] == 5:
                mesh = TetMesh(nodes, elems[:,1:])
                fem  = Solver(mesh)
            stiffness = fem.stiffness.toarray()
            # mass = fem.mass.toarray()
            np.fill_diagonal(stiffness, 0)
            rows, cols = np.nonzero(stiffness)
            A = np.column_stack((rows, cols))
            B = stiffness[rows, cols]
            edge_index  = torch.tensor(A.T, dtype=torch.long).to(device)
            edge_weight = torch.tensor(B, dtype=torch.float).to(device)  
            edge_weight = edge_weight / edge_weight.max()
        else:
            model_type = 'S2NO_spec'

        self.blocks = nn.ModuleList([S2NO_Layer(
                                                    model_type = model_type,
                                                    lbo_bases = lbo_bases,
                                                    lbo_inver = lbo_inver,
                                                    num_channels = num_channels,
                                                    num_heads    = num_heads,
                                                    num_modes    = num_lbos,
                                                    dropout      = 0.0,
                                                    edge_index  = edge_index,
                                                    edge_weight = edge_weight,
                                                    act='gelu',
                                                    mlp_ratio=1,
                                                    kernelsize = 3
                                                ) for _ in range(n_layers)])
            
        if args['initialize_weights'] == True:
            self.initialize_weights()
        if self.placeholder == True:
            self.placeholder_para = nn.Parameter((1 / (num_channels)) * torch.rand(num_channels, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        if self.placeholder == True:        
            x = self.fc0(x) + self.placeholder_para
        else:
            x = self.fc0(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        # x = self.mlp(self.ln(x))
        return x


class S2NO_Layer(nn.Module):
    def __init__(
                    self,
                    model_type,
                    lbo_bases,
                    lbo_inver,
                    num_channels: int,
                    num_heads: int,
                    num_modes: int,
                    dropout: float,
                    edge_index,
                    edge_weight,
                    act='gelu',
                    mlp_ratio=1,
                    kernelsize = 3
                ):
        super(S2NO_Layer, self).__init__()
        
        self.ln_1 = nn.LayerNorm(num_channels)
        self.Conv = Convolution_block(
                                        model_type   = model_type,
                                        num_channels = num_channels, 
                                        num_heads = num_heads,  
                                        num_modes = num_modes, 
                                        edge_index  = edge_index,
                                        edge_weight = edge_weight,
                                        kernelsize = kernelsize,
                                        dropout = dropout)
        
        self.ln_2 = nn.LayerNorm(num_channels)
        self.mlp  = MLP(num_channels, num_channels * mlp_ratio, num_channels, n_layers = 0, res=False, act=act)
        self.lbo_bases = lbo_bases
        self.lbo_inver = lbo_inver
        
    def forward(self, fx):
        fx = self.Conv(self.ln_1(fx), self.lbo_bases, self.lbo_inver) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class Convolution_block(nn.Module):
    def __init__(self, 
                 model_type,
                 num_channels, 
                 num_heads = 8,  
                 num_modes = 64, 
                 edge_index  = None,
                 edge_weight = None,
                 kernelsize = 3,
                 dropout = 0.0): 
        super().__init__()
        
        self.SpectralConv = Spectral_Conv(  num_channels = num_channels, 
                                            num_heads = num_heads,  
                                            num_modes = num_modes, 
                                            kernelsize = kernelsize,
                                            dropout = dropout)
        self.local = True
        if model_type == 'S2NO':
            self.norm = nn.LayerNorm(num_channels)
            self.SpatialConv = Spatial_Conv(num_channels = num_channels,  
                                              edge_index = edge_index)
        elif 'spec' in model_type:
            self.local = None
        else:
            raise ValueError("Please check 'model_type' !")    
        
    def forward(self, x, LBO_MATRIX, LBO_INVERSE ):
        #  x : B N C
        x1 = self.SpectralConv(x, LBO_MATRIX, LBO_INVERSE) 
        if self.local == None:
            return x1
        else:
            x1 = self.norm (x1)
            x2 = self.SpatialConv(x)
            x = x1 + x2
        return x
    

class Spectral_Conv(nn.Module):
    def __init__(self, 
                 num_channels, 
                 num_heads = 8,  
                 num_modes = 64, 
                 kernelsize = 3,
                 dropout = 0.0): 
        super().__init__()
        
        if num_channels % num_heads == 0:
            self.project = True
            
        dim_head = num_channels // num_heads
        inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.heads = num_heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)
        self.in_project_fx     = nn.Conv1d(num_channels, inner_dim, kernelsize, 1, kernelsize//2 )
        self.mlp_trans_weights = nn.Parameter( torch.empty((dim_head, dim_head)) )
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))
        self.layernorm  = nn.LayerNorm( (num_modes, dim_head ) )
        self.to_out     = nn.Sequential( nn.Linear(inner_dim, num_channels), nn.Dropout(dropout) )
        
    def forward(self, x, LBO_MATRIX, LBO_INVERSE):
        
        B, N, C = x.shape
        x = x.permute(0, 2, 1).contiguous()  # B C N
        fx_mid = self.in_project_fx(x).permute(0, 2, 1).contiguous() \
                 .reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N D, HD~C
        spectral_feature = LBO_INVERSE @ fx_mid # B H K D
        
        bsize, hsize, ksize, dsize = spectral_feature.shape
        spectral_feature = self.layernorm(spectral_feature.reshape( -1, ksize, dsize )).reshape( bsize, hsize, ksize, dsize )
        out_spectral_feature = torch.einsum("bhgi,io->bhgo", spectral_feature, self.mlp_trans_weights)
        
        out_x = LBO_MATRIX @ out_spectral_feature
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        
        return self.to_out(out_x)


class Spatial_Conv(nn.Module):
    
    def __init__ (self, 
                  num_channels, 
                  edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.input = nn.Linear(num_channels, num_channels)
        self.gw = nn.Linear(num_channels, num_channels)
        self.out = nn.Linear(num_channels, num_channels)
        self.Graph_conv1 = gnn.ARMAConv(num_channels, num_channels, num_stacks=1, num_layers=1, shared_weights=False)
        # self.Graph_conv2 = gnn.ARMAConv(num_channels, num_channels, num_stacks=1, num_layers=1, shared_weights=False)
        self.Sigmoid  = nn.Sigmoid()
        self.input_gate = nn.Linear(num_channels, num_channels)
        self.output_gate = nn.Linear(num_channels, num_channels)
        
    def forward(self, x):
        input_gate  = self.Sigmoid(self.input_gate(x))
        output_gate = self.Sigmoid(self.output_gate(x))
        x = input_gate * x
        x = F.gelu(self.input(x))
        x = self.Graph_conv1(x, self.edge_index)
        x = F.gelu(self.gw(x))
        # x = self.Graph_conv2(x, self.edge_index)
        x = self.out(x)
        x = output_gate * x
        return x


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 
              'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}
              
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) 
                                      for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x