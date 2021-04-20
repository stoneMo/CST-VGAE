import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
import random

SEED = 2020

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 19)


class Model(nn.Module):
    r"""Graph Autoenconder (GAE).

    Args:
        in_channels (int): Number of channels in the input data
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        predict_frames (int): Number of forecasted frames for the long-term head pose forecasting task
        embedd_dim (int): Size of the dimensionality of the latent embeddings 
        device (str): 'cpu' or 'cuda'
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, C_{in})`
        - Output: #head pose (N, T, 1, 3), landmark (N, T, 19, 2), embeddings (N, embedd_dim)
            :math:`(N, T, 1, 3) & (N, T, 19, 2)` where
            :math:`N` is a batch size,
            :math:`T` is a number of forecasted frames for the long-term head pose forecasting task

    """

    def __init__(self, in_channels,
                 edge_importance_weighting, predict_frames=5, embedd_dim=64, device='cpu', **kwargs):
        super().__init__()
        torch.manual_seed(SEED)
        random.seed(2020)
        # load graph_encoder
        self.graph = Graph(layout='FAN_19', strategy='uniform')  #(1, 19, 19)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False, device=device)
        self.register_buffer('A', A)
        
        # load graph_decoder
        self.graph_de = Graph(layout='FAN_19', strategy='spatial')  #(1, 19, 19)
        A_d = torch.tensor(self.graph_de.A, dtype=torch.float32, requires_grad=False, device=device)
        self.register_buffer('A_d', A_d)

        # build networks
        spatial_kernel_size = A.size(0) 
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size) #(3, 1)
        self.data_bn_en = nn.BatchNorm1d(in_channels * A.size(1))     # 2 * 19
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_encoder = nn.ModuleList((

            # --------------------------------------------------encoder -----------------------------------------
            st_gcn(in_channels, embedd_dim //2, kernel_size, 1, residual=False, **kwargs0),  #(N, 16, 5, 19) 
            
            st_gcn(embedd_dim //2,  embedd_dim //2,  kernel_size, 1, **kwargs),
            st_gcn(embedd_dim //2,  embedd_dim //2,  kernel_size, 1, **kwargs),
            st_gcn(embedd_dim //2,  embedd_dim, kernel_size, 2, **kwargs),             #(N, 32, 3, 19) 
            st_gcn(embedd_dim, embedd_dim, kernel_size, 1, **kwargs),
            st_gcn(embedd_dim, embedd_dim, kernel_size, 1, **kwargs),
            st_gcn(embedd_dim, embedd_dim *2, kernel_size, 2, **kwargs),                #(N, 64, 2, 19) 
            st_gcn(embedd_dim *2, embedd_dim *2, kernel_size, 1, **kwargs), 
            st_gcn(embedd_dim *2, embedd_dim *2, kernel_size, 1, **kwargs),
            st_gcn(embedd_dim *2, embedd_dim *4, kernel_size, 2, **kwargs),              #(N, 128, 1, 19) 
            st_gcn(embedd_dim *4, embedd_dim *4, kernel_size, 1, **kwargs)
        
        ))

        self.flat = Flatten()
        self.fc1 = nn.Linear(embedd_dim *4 *A.size(1), embedd_dim)       # for self.mean
        self.fc2 = nn.Linear(embedd_dim *4 *A.size(1), embedd_dim)       # for self.logvar
        self.unflat = UnFlatten(embedd_dim)
        
        self.fc3 = nn.Sequential(
            nn.Linear(embedd_dim, embedd_dim *A.size(1)),
            nn.PReLU()
        )
        
        self.data_bn_decoder = nn.BatchNorm1d(embedd_dim *A.size(1))       # embedd_dim * 19
        self.st_gcn_decoder_land = nn.ModuleList((

            # --------------------------------------------------decoder -----------------------------------------
            st_gcn(embedd_dim, embedd_dim, kernel_size, 1, **kwargs),             #(N, 32, 1, 19) 
            st_gcn(embedd_dim, embedd_dim //2, (2, spatial_kernel_size), 1, **kwargs),  #(N, 16, 2, 19) 
            st_gcn(embedd_dim //2, embedd_dim //2, kernel_size, 1, **kwargs),
            st_gcn(embedd_dim //2, embedd_dim //4,  (3, spatial_kernel_size), 1, decoder_layer=True, **kwargs),  #(N, 8, 4, 19) 
            st_gcn(embedd_dim //4, embedd_dim //4, kernel_size, 1, **kwargs),
            st_gcn(embedd_dim //4,  2,   (2, spatial_kernel_size), 1, last_layer=True, **kwargs)   #(N, 2, 5, 19) 
            # -------------------------------------------------------------------------------------------------------
        ))
        
        self.land_layer = nn.Sequential(
            nn.Linear(2 * 5 *self.A.size(1), 2*19*predict_frames)
        )
        
        self.data_bn = nn.BatchNorm1d(2 *A.size(1)) 
        
        self.st_gcn_networks = nn.ModuleList((

            # --------------------------------------------------decoder -----------------------------------------
            st_gcn(2, 64, (1, 3), 1, residual=False, **kwargs0),
            st_gcn(64, 64, (1, 3), 1, **kwargs),
            st_gcn(64, 64, (1, 3), 1, **kwargs),
            st_gcn(64, 64, (1, 3), 1, **kwargs),
            st_gcn(64, 128, (1, 3), 2, **kwargs),
#             st_gcn(128, 128, (1, 3), 1, **kwargs),
            # -------------------------------------------------------------------------------------------------------
        ))
        
        # fcn for prediction
        self.fcn = nn.Conv2d(128, 256, kernel_size=1)
       
        # linear layer
        self.linear = nn.Linear(256, 3)
    
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_encoder = nn.ParameterList([          #encoder graph edge 
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_encoder
            ])

            self.edge_importance_mean = nn.Parameter(torch.ones(self.A.size()))
            self.edge_importance_std = nn.Parameter(torch.ones(self.A.size()))

            self.edge_importance_decoder_land = nn.ParameterList([          #decoder_land graph edge 
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_decoder_land
            ])
            
            self.edge_importance = nn.ParameterList([          #decoder_pose graph edge 
                nn.Parameter(torch.ones(self.A_d.size()))
                for i in self.st_gcn_networks
            ])

        else:
            self.edge_importance = [1] * len(self.st_gcn_encoder)
                   

    def encode(self, x):

        x = x[:,:5,:19,:2]    #(N, T, 19, C)

        # data normalization
        N, T, V, C = x.size()  # input tensor：(N, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn_en(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)  --> (N, 2, 5, 19)
       
        # forward
        for gcn, importance in zip(self.st_gcn_encoder, self.edge_importance_encoder):
            x, _ = gcn(x, self.A * importance)  # (N, C, T=2, V=19)

        # print(x.shape)

        x = self.flat(x)

        self.mean = self.fc1(x)
        sampled_z = self.mean

        return sampled_z

    def decode_land(self, z):
        
        z = self.fc3(z)
        z = self.unflat(z)
        
        N, C, T, V = z.size()  # 输入tensor：(N, C, T, V)
        z = z.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        z = z.view(N, V * C, T)
        z = self.data_bn_decoder(z)
        z = z.view(N, V, C, T)
        z = z.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)  --> (N, 64, 1, 21)
        
        z_land = z     #(N, 64, 1, 21)
        #print(z.shape)

        for gcn, importance in zip(self.st_gcn_decoder_land, self.edge_importance_decoder_land):
            z_land, A = gcn(z_land, self.A * importance)  # (N, C=2, T=5, V=19)

        z_land = z_land.view(N, -1)
        y_pred = self.land_layer(z_land)             #(N, 2*19)
        
        y_pred = y_pred.view(N, -1, 19, 2)        

        return y_pred, A

    def decode_pose(self, z):
        
        
        N, V, C = z.size()  # 输入tensor：(N, V, C)
        T = 1
        z = z.view(N, V * C, T)
        z = self.data_bn(z)
        z = z.view(N, V, C, T)
        z = z.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)  --> (N, 64, 1, 21)
        
        z_pose = z      #(N, 64, 1, 21)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            z_pose, A = gcn(z_pose, self.A_d * importance)  # (N, C=128, T=1, V=19)
                        
        y = F.avg_pool2d(z_pose, z_pose.size()[2:])    
        y = y.view(N, -1, 1, 1)      #(N, 128, 1, 1)
       
        # prediction 
        y = self.fcn(y)           #(N, 256, 1, 1)
        
        y = y.view(y.size(0), -1)

        y_pose = self.linear(y)     #(N, 3)
        y_pose = y_pose.view(N, -1, 1, 3)
        
        return y_pose, A

    def forward(self, x, predict_frames):
        
        z = self.encode(x)                   # z: (N, 64, 1, 19)
        #print(f'Z: {z.shape}')
        
        y_land, A_pred = self.decode_land(z)
        pose_result = []
        for i in range(predict_frames):
            y_pose, _ = self.decode_pose(y_land[:,i,:,:])
            pose_result.append(y_pose)
         
        pose_result = torch.cat(pose_result, dim=1)
        
        return pose_result, y_land, z



class st_gcn(nn.Module):
    r"""Spatial temporal graph convolution network (ST-GCN).

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 last_layer=False,
                 decoder_layer=False,
                 re_layer=False):
        super().__init__()

        assert len(kernel_size) == 2
        #print(kernel_size[0])
        if kernel_size[0] == 3 and not decoder_layer:
            padding = ((kernel_size[0] - 1) // 2, 0)
            stride_res = 2
        elif re_layer:
            padding = (0, 0)
            stride_res = 2
        elif last_layer:
            padding = ((kernel_size[0] + 1) //2, 0) 
            stride_res = 4
        elif not last_layer and kernel_size[0] != 1: 
            padding = ((kernel_size[0] + 1) //2, 0) 
            stride_res = 2
        elif kernel_size[0] == 1:
            padding = ((kernel_size[0] - 1) // 2, 0)
            stride_res = stride

        # GCN
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        # TCN
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else: 
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride_res, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
        self.kernel_size = kernel_size

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        tcn = self.tcn(x)
        x = tcn + res

        return self.relu(x), A
