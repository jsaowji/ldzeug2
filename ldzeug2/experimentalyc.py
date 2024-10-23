import torch
from torch import nn
import numpy as np
from .compact import compact

# based on the one from neosr
class experimental(nn.Module):
    def __init__(self,num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=1, kernel_size=3, act_type='prelu',bias=True,**kwargs):
        super(experimental, self).__init__()
        self.compa = compact(num_in_ch, num_out_ch, num_feat, num_conv, upscale, kernel_size, act_type,bias)
        
    def forward(self, x):
        out = self.compa(x)
        #return out
        return x[:,[0],:,:] + out

