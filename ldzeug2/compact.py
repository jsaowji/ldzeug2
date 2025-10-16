import torch
from torch import nn
import numpy as np

# based on the one from neosr
class compact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=1, kernel_size=3, act_type='prelu',bias=True, addback=False,**kwargs):
        super(compact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.addback = addback

        strid = 1
        padd = "same"

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, kernel_size, strid, padd,bias=bias))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, kernel_size, strid, padd,bias=bias))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch *
                         upscale * upscale, kernel_size, strid, padd,bias=bias))

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        if self.addback:
            out += x
        return out

from .stackable import Stackable,StackableManager

# bias not implemented
def compact_to_expr(model: compact) -> str:
    work = [Stackable.plane(f"src{a}") for a in range(70)]
    output_txt = ""
    for i,a in enumerate(model.body):
        #print(a)
        if isinstance(a,nn.Conv2d):
            output_txt += "work = [\n"
            for out_ch in range(a.out_channels):
                cout = Stackable.const(0)
                for in_ch in range(a.in_channels):
                    matr = a.weight[out_ch,in_ch,:,:].detach().numpy()

                    ll = Stackable.const(0)
                    for j in range(a.kernel_size[0]):
                        for k in range(a.kernel_size[1]):
                            ll += work[in_ch][ -(a.kernel_size[0]-1)//2 + k, -(a.kernel_size[1]-1)//2 + j   ] * (matr[j,k])
                    cout += ll
                expr = cout.eval({})

                output_txt += f'core.akarin.Expr(work,"{expr}"),\n'
            output_txt += "]\n"
        elif isinstance(a, nn.PReLU):
            output_txt += "work = [\n"
            for in_out_ch in range(a.num_parameters):
                x = work[in_out_ch]
                expr = (x >= 0).iftrue(x, x * float(a.weight[in_out_ch].detach().numpy()))
                expr  = expr.eval({})
                output_txt += f'core.akarin.Expr(work,"{expr}"),\n'
            output_txt  += "]\n"
        else:
            assert False
    import textwrap

    output_txt = textwrap.indent(output_txt + "\nreturn work", '    ')
    output_txt = "from vstools import core\ndef apply_model(work):\n" + output_txt 
    return output_txt
