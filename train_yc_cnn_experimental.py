# ruff: noqa
#%%%
%load_ext autoreload
%autoreload 2
#%%
import vapoursynth as vs
core = vs.core


core.std.LoadPlugin("/usr/lib/vapoursynth/libvsrawsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libakarin.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/bestsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libresize2.so")
#core.std.LoadPlugin("/usr/lib/vapoursynth/libfpng.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libcolorbars.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libvslsmashsource.so")
#%%
import json
import functools
from pathlib import Path
from vstools import vs, initialize_clip, set_output, depth, padder, Matrix, Transfer, Primaries, ColorRange,split,join
import random
import numpy as np
from matplotlib import pyplot as plt
from ldzeug2.vsnn import fill_train_buffer,cut_to_rndm_frames,cache_all_frames,interleave_clips,load_random_train_frame_from_vnode
from ldzeug2.colorencoder import modulate_fields
import numpy as np
import matplotlib.pyplot as plt


exec(open("clips.py","rt").read())
clips: list[vs.VideoNode] = out
og = interleave_clips(clips)
og = og

remaped = cut_to_rndm_frames(og,30)

modulated_in  = modulate_fields(remaped.resize.Bilinear(760,486,format=vs.YUV444P16).std.SeparateFields(tff=True))
modulated_out = modulate_fields(remaped.resize.Bilinear(760*2,486*2,format=vs.YUV444P16).std.SeparateFields(tff=True))

train_input  = join([modulated_in.tbc_out,modulated_in.i_carier,modulated_in.q_carier]).std.DoubleWeave(tff=True)[::2]
train_output = join([modulated_out.luma_out,modulated_out.i_hp,modulated_out.q_hp]).std.DoubleWeave(tff=True)[::2]

print("caching")
cache_all_frames(train_input)
cache_all_frames(train_output)
#ff = lvsfunc.get_random_frames(train_input,dur=0.00000001)
#lvsfunc.export_frames(train_input,filename="/tmp/lq/%d.png",frames=ff)
#%%
#from ldzeug2.colorcnnv1 import FullModel
from ldzeug2.experimentalyc import FullModelExperimental
import torch
torch.set_default_device('cuda')

mdlpth = "/tmp/rr46.pth"

model = FullModelExperimental(num_feat=64,num_conv=16,upscale=2)
model.load_state_dict(torch.load(mdlpth))



i = 0

init_lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr,betas=[0.9, 0.99])
#optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.0)
loss_fn = torch.nn.L1Loss()


#%%
from ldzeug2.vsnn import fill_train_buffer_ex
lq_size = 32
upscale = 2
gt_size = lq_size * upscale
batch_cnt = 6


tnsrss_in  = torch.ones((batch_cnt,3,lq_size,lq_size))
tnsrss_out = torch.ones((batch_cnt,3,gt_size,gt_size))

while True:
    frame_num, f_width,f_height,hq,lq = load_random_train_frame_from_vnode(train_output,train_input)
    f_width_lq = f_width // upscale
    f_height_lq = f_height // upscale
    lq = np.expand_dims(lq,0)
    hq = np.expand_dims(hq,0)
    
    #add extra noise
    #lq += np.random.normal(scale=0.025,size=lq.shape)
        

    lq = torch.Tensor(lq).cuda()
    hq = torch.Tensor(hq).cuda()
    
    fill_train_buffer_ex(tnsrss_in,tnsrss_out,lq,hq,f_width_lq,f_height_lq,lq_size,upscale,batch_cnt)

    optimizer.zero_grad()
    model_outputs = model(tnsrss_in)
    loss = loss_fn(model_outputs, tnsrss_out)
    
    
    loss.backward()
    optimizer.step()
    
    save_it = 1000
    if i % save_it == 0:
        mdll = model_outputs.detach().cpu().numpy()[0,0]
        hql = tnsrss_out.detach().cpu().numpy()[0,0]
        plt.subplot(241)
        plt.imshow(mdll)
        plt.title(f"model luma")
        #plt.show()
        
        plt.subplot(242)
        plt.imshow(hql)
        plt.title("hq luma")
        #plt.show()
        
        plt.subplot(243)
        plt.imshow(tnsrss_in.detach().cpu().numpy()[0,0])
        plt.title("lq luma")
        
        plt.subplot(244)
        plt.imshow((torch.abs(model_outputs - tnsrss_out)).detach().cpu().numpy()[0,0])
        plt.title("err luma")
        
        
        
        
        mdll = model_outputs.detach().cpu().numpy()[0,1]
        hql = tnsrss_out.detach().cpu().numpy()[0,1]
        plt.subplot(245)
        plt.imshow(mdll)
        plt.title(f"model i")
        #plt.show()
        
        plt.subplot(246)
        plt.imshow(hql)
        plt.title("hq i")
        #plt.show()
        
        plt.subplot(247)
        plt.imshow(tnsrss_in.detach().cpu().numpy()[0,1])
        plt.title("i mlt")
        
        plt.subplot(248)
        plt.imshow((torch.abs(model_outputs - tnsrss_out)).detach().cpu().numpy()[0,1])
        plt.title("err i")
        
        
        
        
        plt.show()
        print(i,losnow)

        #lq = frameto_numpy(train_input,a)

    if (i % save_it) == 0:
        torch.save(model.state_dict(),mdlpth)
    losnow = loss.detach().cpu().numpy()
    #print(i,losnow)
    i += 1
    
# %%
