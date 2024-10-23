from .colorencoder import annon_phase_frames_to_fields
from .colordecoder import uv_from_iq,to_yuv

from .comb_consts import COLOR_CARIER_FREQ_FLT, CombConsts
from .stackable import *
from vstools import join,split,core,vs
import math
from .utils import get_model_path

import torch
from .compact import compact
from torch import nn

class FullModel(torch.nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.small_model = compact(1,1,num_feat=32, num_conv=12, bias=False)
        self.big_model   = compact(3,3,num_feat=16, num_conv=6, bias=False)
    
    def forward(self,x):
        inbuf         =  x[:,0,:,:].unsqueeze(1)
        i_mlt         =  x[:,1,:,:].unsqueeze(1)
        q_mlt         =  x[:,2,:,:].unsqueeze(1)

        so = self.small_model(inbuf)
        alt_chroma = inbuf - so

        midbuf = torch.stack([
            (so                )[:,0,:,:],
            (alt_chroma * i_mlt)[:,0,:,:],
            (alt_chroma * q_mlt)[:,0,:,:]
        ]).permute((1,0,2,3))

        outbuf = self.big_model(midbuf)
        outbuf[:,0,:,:] += so[:,0,:,:]
        
        return outbuf

class FullModel2(torch.nn.Module):
    def __init__(self,num_feat=64,num_conv=16):
        super(FullModel2, self).__init__()
        num_in_ch = 1
        num_out_ch = 3
        kernel_size = 3
        strid = 1
        padd = "same"
        bias = True

        self.num_feat = num_feat
        self.num_conv = num_conv

        self.body1 = nn.ModuleList()
        self.body2 = nn.ModuleList()
        self.body3 = nn.ModuleList()

        self.body1.append(nn.Conv2d(num_in_ch, num_feat, kernel_size, strid, padd,bias=bias))
        self.body1.append(nn.PReLU(num_parameters=num_feat))

        self.params1 = torch.nn.parameter.Parameter(torch.randn(num_feat))
        self.params2 = torch.nn.parameter.Parameter(torch.randn(num_feat))
            
        for _ in range(num_conv):
            self.body2.append(nn.Conv2d(num_feat, num_feat, kernel_size, strid, padd,bias=bias))
            self.body2.append(nn.PReLU(num_parameters=num_feat))

        self.body3.append(nn.Conv2d(num_feat, num_out_ch, kernel_size, strid, padd,bias=bias))

    
    def forward(self,x):
        batch_cnt,planes,w,h = x.shape
        
        inbuf         =  x[:,0,:,:].unsqueeze(1)
        i_mlt         =  x[:,1,:,:].unsqueeze(1)
        q_mlt         =  x[:,2,:,:].unsqueeze(1)

        out = inbuf
        for i in range(0, len(self.body1)):
            out = self.body1[i](out)
        
        for i in range(self.num_conv):
            out = self.body2[i](out)
            if i == self.num_conv // 2:
                p = self.params1.reshape((1,self.num_feat,1,1))
                out = ((1.0 - p) * out) + ( (p) * (out * i_mlt))
            
                p = self.params2.reshape((1,self.num_feat,1,1))
                out = ((1.0 - p) * out) + ( (p) * (out * q_mlt))

        for i in range(0, len(self.body3)):
            out = self.body3[i](out)

        return out