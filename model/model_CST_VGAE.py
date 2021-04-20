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
    def __init__(self, size, A_size):
        super().__init__()
        self.size = size
        self.A_size = A_size
        
    
    def forward(self, input):
        return input.view(input.size(0), self.size, 1, self.A_size)

def st_gcn_encoder(in_channels, embedd_dim, kernel_size, kwargs0, kwargs):

    st_gcn_encoder = nn.ModuleList((

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
    return st_gcn_encoder

class TemporalSelfAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, num_head, dim_input, dim_query, dim_key, dim_value):
        super().__init__()

        self.num_head = num_head
        self.dim_query = dim_query
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.W_q = nn.Linear(dim_input, num_head * dim_query, bias=False)
        self.W_k = nn.Linear(dim_input, num_head * dim_key, bias=False)
        self.W_v = nn.Linear(dim_input, num_head * dim_value, bias=False)
        self.W_o = nn.Linear(num_head * dim_value, dim_input, bias=False)

    def forward(self, query, key, value):

        # d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size = query.size(0)
        temporal_query, temporal_key, temporal_value = query.size(1), key.size(1), value.size(1)
        residual = query

        query = self.W_q(query).view(batch_size, temporal_query, self.num_head, self.dim_query)
        key = self.W_k(key).view(batch_size, temporal_key, self.num_head, self.dim_key)
        value = self.W_v(value).view(batch_size, temporal_value, self.num_head, self.dim_value)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        attention = torch.matmul(query / (self.dim_key**0.5), key.transpose(2, 3))
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, value)

        output = output.transpose(1, 2).contiguous().view(batch_size, temporal_query, -1)
        output = self.W_o(output)
        output += residual

        output = output.view(batch_size, -1)

        return output, attention

class Model(nn.Module):
    r"""Conditional Spatial-Temporal Variational Graph Autoenconder (CST-VGAE).

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
        self.graph_y = Graph(layout='FAN_21', strategy='uniform')  #(1, 21, 21)
        A_y = torch.tensor(self.graph_y.A, dtype=torch.float32, requires_grad=False, device=device)
        self.register_buffer('A_y', A_y)

        self.graph_p = Graph(layout='FAN_21', strategy='uniform')  #(1, 21, 21)
        A_p = torch.tensor(self.graph_p.A, dtype=torch.float32, requires_grad=False, device=device)
        self.register_buffer('A_p', A_p)

        self.graph_r = Graph(layout='FAN_21', strategy='uniform')  #(1, 21, 21)
        A_r = torch.tensor(self.graph_r.A, dtype=torch.float32, requires_grad=False, device=device)
        self.register_buffer('A_r', A_r)

        # print("A_y", A_y.size())
        # print("A_p", A_p.size())
        # print("A_r", A_r.size())

        if A_y.size() == A_p.size() and A_p.size() == A_r.size():
            A = A_y      # A just for representing these three graphs' size
        else: 
            raise NameError('all A_y, A_p, A_r should have the same size!')
        
        # load graph_decoder
        self.graph_de = Graph(layout='FAN_19', strategy='spatial')  #(1, 19, 19)
        A_d = torch.tensor(self.graph_de.A, dtype=torch.float32, requires_grad=False, device=device)
        self.register_buffer('A_d', A_d)

        # build networks
        spatial_kernel_size = A_y.size(0)       # should be 1
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size) #(3, 1)
        self.data_bn_en = nn.BatchNorm1d(in_channels * A_y.size(1))     # 2 * 21
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.fixed_past_frame = 5

        self.st_gcn_encoder_y = st_gcn_encoder(in_channels, embedd_dim, kernel_size, kwargs0, kwargs)
        self.st_gcn_encoder_p = st_gcn_encoder(in_channels, embedd_dim, kernel_size, kwargs0, kwargs)
        self.st_gcn_encoder_r = st_gcn_encoder(in_channels, embedd_dim, kernel_size, kwargs0, kwargs)
        
        self.flat = Flatten() 
        self.fc1_y = nn.Linear(embedd_dim *4 *A_y.size(1), embedd_dim)       # for self.mean_y
        self.fc2_y = nn.Linear(embedd_dim *4 *A_y.size(1), embedd_dim)       # for self.logvar_y

        self.fc1_p = nn.Linear(embedd_dim *4 *A_p.size(1), embedd_dim)       # for self.mean_p
        self.fc2_p = nn.Linear(embedd_dim *4 *A_p.size(1), embedd_dim)       # for self.logvar_p

        self.fc1_r = nn.Linear(embedd_dim *4 *A_r.size(1), embedd_dim)       # for self.mean_r
        self.fc2_r = nn.Linear(embedd_dim *4 *A_r.size(1), embedd_dim)       # for self.logvar_r

        self.gaussian_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) 
            for i in range(3)
        ])

        self.unflat = UnFlatten(embedd_dim, A.size(1))
        
        self.fc3 = nn.Sequential(
            nn.Linear(embedd_dim + 4*self.fixed_past_frame, embedd_dim *A.size(1)),
            nn.PReLU()
        )

        #--------------------------Temporal Self-attention Module---------------------------------

        self.dim_attention = 64
        self.num_head = 8
        
        self.slf_c_attention = TemporalSelfAttention(
                            num_head = self.num_head, 
                            dim_input= 4,
                            dim_query = self.dim_attention,
                            dim_key = self.dim_attention,
                            dim_value = self.dim_attention
        )

        # ------------------------------------------------------------------------------------
        
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
            nn.Linear(2 * 5 *self.A_y.size(1), 2*19*self.fixed_past_frame)
        )
        
        self.data_bn = nn.BatchNorm1d(2 *A_d.size(1)) 
        
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
            self.edge_importance_encoder_y = nn.ParameterList([          #encoder graph_y edge 
                nn.Parameter(torch.ones(self.A_y.size()))
                for i in self.st_gcn_encoder_y
            ])

            self.edge_importance_encoder_p = nn.ParameterList([          #encoder graph_p edge 
                nn.Parameter(torch.ones(self.A_p.size()))
                for i in self.st_gcn_encoder_p
            ])

            self.edge_importance_encoder_r = nn.ParameterList([          #encoder graph_r edge 
                nn.Parameter(torch.ones(self.A_r.size()))
                for i in self.st_gcn_encoder_r
            ])
            
            self.edge_importance_decoder_land = nn.ParameterList([          #decoder_land graph edge 
                nn.Parameter(torch.ones(self.A_y.size()))
                for i in self.st_gcn_decoder_land
            ])
            
            self.edge_importance = nn.ParameterList([          #decoder_pose graph edge 
                nn.Parameter(torch.ones(self.A_d.size()))
                for i in self.st_gcn_networks
            ])

            
        else:
            self.edge_importance = [1] * len(self.st_gcn_encoder)
                   

    def encode(self, x):

        x = x[:,:5,:21,:]    #(N, T, 21, C)

        # data normalization
        N, T, V, C = x.size()  # input tensor：(N, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn_en(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)  --> (N, 2, 5, 19)

        x_y = x
        x_p = x
        x_r = x

        # forward_y
        for gcn, importance in zip(self.st_gcn_encoder_y, self.edge_importance_encoder_y):
            x_y, _ = gcn(x_y, self.A_y * importance)  # (N, C, T=2, V=19)

        # forward_p
        for gcn, importance in zip(self.st_gcn_encoder_p, self.edge_importance_encoder_p):
            x_p, _ = gcn(x_p, self.A_p * importance)  # (N, C, T=2, V=19)

        # forward_r
        for gcn, importance in zip(self.st_gcn_encoder_r, self.edge_importance_encoder_r):
            x_r, _ = gcn(x_r, self.A_r * importance)  # (N, C, T=2, V=19)


        x_y = self.flat(x_y)
        self.mean_y = self.fc1_y(x_y)
        self.logvar_y = self.fc2_y(x_y)

        x_p = self.flat(x_p)
        self.mean_p = self.fc1_p(x_p)
        self.logvar_p = self.fc2_p(x_p)

        x_r = self.flat(x_r)
        self.mean_r = self.fc1_r(x_r)
        self.logvar_r = self.fc2_r(x_r)

        # for Gaussian_y 
        gaussian_noise_y = torch.randn(self.mean_y.size(), device=self.mean_y.device)
        sampled_z_y = gaussian_noise_y*torch.exp(self.logvar_y.mul(0.5)) + self.mean_y  #(N, C) --> (N, 32)

        # for Gaussian_p
        gaussian_noise_p = torch.randn(self.mean_p.size(), device=self.mean_p.device)
        sampled_z_p = gaussian_noise_p*torch.exp(self.logvar_p.mul(0.5)) + self.mean_p  #(N, C) --> (N, 32)

        # for Gaussian_r
        gaussian_noise_r = torch.randn(self.mean_r.size(), device=self.mean_r.device)
        sampled_z_r = gaussian_noise_r*torch.exp(self.logvar_r.mul(0.5)) + self.mean_r  #(N, C) --> (N, 32)

        # Total GMM
        sampled_z = self.gaussian_weights[0]*sampled_z_y + self.gaussian_weights[1]*sampled_z_p + self.gaussian_weights[2]*sampled_z_r
       
        return sampled_z

    def decode_land(self, z):
        
        z = self.fc3(z)
        z = self.unflat(z)
        
        N, C, T, V = z.size()  # 输入tensor：(N, C, T, V)
        z = z.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        z = z.view(N, V * C, T)
        z = self.data_bn_decoder(z)
        z = z.view(N, V, C, T)
        z = z.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)  
        
        z_land = z     #(N, 64, 1, 21)

        for gcn, importance in zip(self.st_gcn_decoder_land, self.edge_importance_decoder_land):
            z_land, A = gcn(z_land, self.A_y * importance)  # (N, C=2, T=5, V=19)

        z_land = z_land.view(N, -1)
        y_pred = self.land_layer(z_land)             #(N, 2*19)
        
        y_pred = y_pred.view(N, 2, -1, 19)

        return y_pred, A

    def decode_pose(self, z):
        
        N, C, T, V = z.size()  # 输入tensor：(N, C, T, V)
        z = z.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        z = z.view(N, V * C, T)
        z = self.data_bn(z)
        z = z.view(N, V, C, T)
        z = z.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V) 
        z_pose = z      #(N, 64, 1, 21)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            z_pose, A = gcn(z_pose, self.A_d * importance)  # (N, C, T, V=19)
                        
        y = F.avg_pool2d(z_pose, z_pose.size()[2:])    
        y = y.view(N, -1, 1, 1)      #(N, 128, 1, 1)
       
        # prediction 
        y = self.fcn(y)           #(N, 256, 1, 1)
        
        y = y.view(y.size(0), -1)

        y_pose = self.linear(y)     #(N, 3)
        y_pose = y_pose.view(N, -1, 1, 3)
        
        return y_pose, A

    def forward(self, x, predict_frames):
        
        bs = x.size(0)
        fixed_past_frame = 5
        e_z = self.encode(x)                   # z: (N, 64)
        c_0 = x[:,:fixed_past_frame,:2,-2:].reshape(bs, -1)      # c: (N, 5, 2, 2)

        num_iteration = predict_frames // fixed_past_frame

        # self-supervised landmarks
        y_land_list = []
        for i in range(num_iteration):
            if i == 0: 
                c_z = c_0
            z = torch.cat((e_z,c_z), dim=-1)
            y_land, A_pred = self.decode_land(z)    #(N, 2, num_pred_frame, 21) 
            y_land_list.append(y_land)
            c_recurr = y_land[:,:,:,-2:].permute(0, 2, 3, 1).contiguous()  # c:(N, 5, 2, 2)
            
            ## Attention(c_{i}, c_{i-1})
            c_future = c_recurr.reshape(bs, fixed_past_frame, -1)  #  future gaze
            c_past = c_z.reshape(bs, fixed_past_frame, -1)           # past gaze
            c_z, attention = self.slf_c_attention(c_past, c_future, c_future)

        y_land = torch.cat(y_land_list, dim=2)
        land_pred = y_land.permute(0, 2, 3, 1).contiguous()  #(N, -1, 21, 2) 

        # generated head poses
        pose_result = []
        for i in range(predict_frames):
            y_pose, _ = self.decode_pose(y_land[:,:,i:i+1,:19])
            pose_result.append(y_pose)
        pose_result = torch.cat(pose_result, dim=1)
        
        return pose_result, land_pred, z

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


