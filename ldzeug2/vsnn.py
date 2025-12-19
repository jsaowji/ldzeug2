from typing import Callable
from vstools import core, initialize_clip, vs
import random
import numpy as np



import json
import functools
from pathlib import Path
from vstools import remap_frames, vs, initialize_clip, set_output, depth, padder, Matrix, Transfer, Primaries, ColorRange,split,join
import random
import numpy as np
from matplotlib import pyplot as plt
#from .vsnn import fill_train_buffer,cut_to_rndm_frames,cache_all_frames, get_rndm_frames,interleave_clips,load_random_train_frame_from_vnode, load_random_train_frame_from_vnode_idx_given, open_all_clips
from .colorencoder import modulate_fields
import numpy as np
import matplotlib.pyplot as plt
import vskernels as vke

from .stackable import *
from enum import Enum
from .compact import compact
from .colorcnn_trch import ColorCNNV1,ColorCNNV2

class ModelType(Enum):
    COMPACT    = 1
    COLORCNNv1 = 2
    COLORCNNv2 = 3

    def get_torch_model(self, **kwargs):
        if self == ModelType.COMPACT:
            return compact(num_in_ch=1, num_out_ch=1, **kwargs)
        elif self == ModelType.COLORCNNv1:
            return ColorCNNV1(**kwargs)
        elif self == ModelType.COLORCNNv2:
            return ColorCNNV2(**kwargs)

def open_all_clips(file="clips.txt") -> list[vs.VideoNode]:
    def vidf(a):
        print(f"Opending {a}")
        return core.bs.VideoSource(a)
    out = [ a.strip() for a in open(file,"rt").readlines()]
    out = [ vidf(a) for a in out]
    out = [ initialize_clip(a.std.SetFieldBased(0).std.AssumeFPS(out[0])).resize.Bicubic(matrix_s="170m") for a in out]
    return out

def build_train_dataset(model:ModelType=ModelType.COMPACT, count=180, on_fields=True, deformations: list[Callable]=[]):
    og = interleave_clips(open_all_clips())
   
    in_kernel  = vke.Bicubic()

    og = in_kernel.scale(og,760,486,format=vs.YUV444P16)
    
    frms = get_rndm_frames(og)
    og_in  = remap_frames(og,frms[:count]).std.SeparateFields(tff=True)

    modulated_in  = modulate_fields(og_in)

    if model == ModelType.COMPACT:
        train_input  = modulated_in.tbc_out
        train_output = modulated_in.luma_out
    elif model == ModelType.COLORCNNv1:
        pass
    elif model == ModelType.COLORCNNv2:
        train_input  = join([
            modulated_in.tbc_out,
            modulated_in.i_carier,
            modulated_in.q_carier,
        ])
        train_output = join([
            modulated_in.luma_out,
            modulated_in.i_lp,
            modulated_in.q_lp,
        ])
    if on_fields:
        pass
    else:
        train_output = train_output.std.DoubleWeave(tff=True)[::2]
        train_input  = train_input.std.DoubleWeave(tff=True)[::2]
    print("Caching")
    cache_all_frames(train_input)
    cache_all_frames(train_output)
    print("Caching done")
    return train_input, train_output
import torch
import os

from ldzeug2.colorcnn_trch import ColorCNNV2

class Training:
    def __init__(self, model_fn, dataset, model_path="/tmp/test.pth"):
        self.model_path = model_path
        self.model_fn = model_fn

        torch.set_default_device('cuda')
        self.train_input, self.train_output = dataset
        model = model_fn()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        self.i = 0
        init_lr = 1e-4
        self.optimizer = torch.optim.Adam(model.parameters(), lr=init_lr,betas=[0.9, 0.99])
        #optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.0)
        self.loss_fn = torch.nn.MSELoss()
        
        self.gt_size = 64
        self.batch_cnt = 12
        self.save_it = 1000
        
        self.channel_count = 0
        self.channel_count = 1
        self.tnsrss_in  = torch.ones((self.batch_cnt, self.channel_count, self.gt_size, self.gt_size))
        self.tnsrss_out = torch.ones((self.batch_cnt, self.channel_count, self.gt_size, self.gt_size))

        self.model = model

    def train(self):
        while True:
            frame_num, f_width,f_height,hq,lq = load_random_train_frame_from_vnode_idx_given((self.i % len(self.train_output) // 2) * 2+0, self.train_output, self.train_input)
            lq = torch.Tensor(np.expand_dims(lq,0)).cuda()
            hq = torch.Tensor(np.expand_dims(hq,0)).cuda()

            fill_train_buffer(self.tnsrss_in, self.tnsrss_out,lq,hq,f_width,f_height,self.gt_size,self.batch_cnt//2,0)
            frame_num, f_width,f_height,hq,lq = load_random_train_frame_from_vnode_idx_given((self.i % len(self.train_output) // 2) * 2+1,self.train_output,self.train_input)
            lq = torch.Tensor(np.expand_dims(lq,0)).cuda()
            hq = torch.Tensor(np.expand_dims(hq,0)).cuda()


            fill_train_buffer(self.tnsrss_in,self.tnsrss_out,lq,hq,f_width,f_height,self.gt_size,self.batch_cnt//2,self.batch_cnt//2)

            self.optimizer.zero_grad()

            model_outputs = self.model(self.tnsrss_in)
            loss = self.loss_fn(model_outputs, self.tnsrss_out)

            loss.backward()
            self.optimizer.step()


            if (self.i % self.save_it) == 0:
                torch.save(self.model.state_dict(), self.model_path)


            losnow = loss.detach().cpu().numpy()
            print(self.i,losnow)

            if self.i % self.save_it == 0:
            #if True:
                mdll = model_outputs.detach().cpu().numpy()[0,0]
                lql = self.tnsrss_in.detach().cpu().numpy()[0,0]
                hql = self.tnsrss_out.detach().cpu().numpy()[0,0]
                
                plt.subplot(131)
                plt.imshow(lql)
                plt.title("lq luma")
                
                plt.subplot(132)
                plt.imshow(hql)
                plt.title("hq luma")
                #plt.show()

                plt.subplot(133)
                plt.imshow(mdll)
                plt.title(f"model luma")
                #plt.show()


                plt.show()

            self.i += 1

def interleave_clips(a: list[vs.VideoNode]):
    smallest_len = 2**64
    for b in a:
        smallest_len = min(smallest_len, len(b))
    return core.std.Interleave([c[:smallest_len] for c in a])


def get_rndm_frames(og):
    return get_rndm_frames_cnt(len(og))


def get_rndm_frames_cnt(cntt):
    frms = list(range(cntt))
    random.shuffle(frms)
    return frms


def cut_to_rndm_frames(og: vs.VideoNode, frmcn) -> vs.VideoNode:
    frms = get_rndm_frames(og)

    from vstools import remap_frames

    return remap_frames(og, frms[:frmcn])


def cache_all_frames(a: vs.VideoNode):
    a.std.SetVideoCache(1, 2 * len(a) + 50, 2 * len(a) + 50)
    for i, a in enumerate(a.frames()):
        print(i, len(a))


def cache_all_frames_vstools(inp: vs.VideoNode) -> vs.VideoNode:
    from vstools import cache_clip

    a = cache_clip(inp, cache_size=len(inp))
    for i, af in enumerate(a.frames()):
        print(i, len(a))
    return a


def get_rndm_crop(width, height, crpp):
    xof = random.randint(0, width)
    yof = random.randint(0, height)

    # xof = xof - max(0,xof-width-crpp)
    # yof = yof - max(0,yof-height-crpp)
    if xof + crpp > width:
        xof = width - crpp
    if yof + crpp > height:
        yof = height - crpp

    yof -= yof % 2
    xof -= xof % 4

    return xof, yof


def frameto_numpy(xx: vs.VideoNode, a: int):
    hqf = xx.get_frame(a)
    hq = np.ones((len(hqf), xx.height, xx.width))

    for abb in range(len(hqf)):
        hq[abb, :, :] = np.asarray(hqf[abb])
    return hq


def load_random_train_frame_from_vnode(
    train_output: vs.VideoNode, train_input: vs.VideoNode
):
    a = random.randint(0, len(train_output) - 1)
    return load_random_train_frame_from_vnode_idx_given(a, train_output, train_input)


def load_random_train_frame_from_vnode_idx_given(
    a: int, train_output: vs.VideoNode, train_input: vs.VideoNode
):
    hq = frameto_numpy(train_output, a)
    lq = frameto_numpy(train_input, a)

    f_height = train_output.height
    f_width = train_output.width
    return a, f_width, f_height, hq, lq


def fill_train_buffer(
    tnsrss_in, tnsrss_out, lq, hq, f_width, f_height, gt_size, batch_cnt, skip=0
):
    for a in range(skip, batch_cnt):
        xof, yof = get_rndm_crop(f_width, f_height, gt_size)
        inputs_cr = lq[:, :, yof : (yof + gt_size), xof : (xof + gt_size)]
        outputs_cr = hq[:, :, yof : (yof + gt_size), xof : (xof + gt_size)]
        tnsrss_in[a, :, :, :] = inputs_cr
        tnsrss_out[a, :, :, :] = outputs_cr


def fill_train_buffer_ex(
    tnsrss_in, tnsrss_out, lq, hq, f_width_lq, f_height_lq, lq_size, upscale, batch_cnt
):
    gt_size = lq_size * upscale
    for a in range(batch_cnt):
        xof, yof = get_rndm_crop(f_width_lq, f_height_lq, lq_size)

        xofu, yofu = xof * upscale, yof * upscale

        inputs_cr = lq[:, :, yof : (yof + lq_size), xof : (xof + lq_size)]
        outputs_cr = hq[:, :, yofu : (yofu + gt_size), xofu : (xofu + gt_size)]
        tnsrss_in[a, :, :, :] = inputs_cr
        tnsrss_out[a, :, :, :] = outputs_cr


def sanity_check_model():
    pass
    # x = torch.empty((1,1,263,910))
    # assert x.shape == model(x).shape
    #
    #
    #
    # for _ in range(100):
    #    import random
    #    w =  random.randint(10,100)
    #    h =  random.randint(10,100)
    #    w -= w % 2
    #    #w -= w % 4
    #
    #    x = torch.empty((1,1,h,w))
    #
    #    aa = model(x)
    #    print(x.shape,aa.shape)
    #    assert x.shape == aa.shape


def torch_model_frame_eval(
    node: vs.VideoNode, output: vs.VideoNode, model
) -> vs.VideoNode:
    def apply_model22(n, f, model):
        import numpy as np
        from torch import Tensor

        frm0 = Tensor(np.array(f[0])).unsqueeze(0)
        out = model(frm0)
        fout = f[1].copy()
        for i in range(len(fout)):
            np.copyto(np.asarray(fout[i]), out.detach().cpu().numpy()[0, i])
        return fout

    return core.std.ModifyFrame(output, [node, output], apply_model22)

