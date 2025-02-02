import vapoursynth as vs
core = vs.core


core.std.LoadPlugin("/usr/lib/vapoursynth/libvsrawsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libakarin.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/bestsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libresize2.so")
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
og = og.resize.Bilinear(760,486,format=vs.YUV444P16).std.SeparateFields(tff=True)

remaped = cut_to_rndm_frames(og,300)

modulated_in = modulate_fields(remaped)

train_output = modulated_in.luma_out
train_input  = modulated_in.tbc_out

print("caching")
cache_all_frames(train_input)
cache_all_frames(train_output)
#ff = lvsfunc.get_random_frames(train_input,dur=0.00000001)
#lvsfunc.export_frames(train_input,filename="/tmp/lq/%d.png",frames=ff)
#%%
from ldzeug2.compact import compact
import torch
import os
torch.set_default_device('cuda')

mdlpth = "/tmp/yc_cnn_smol.pth"

model = compact(num_in_ch=1, num_out_ch=1, num_feat=32, num_conv=14, upscale=1, kernel_size=3, act_type='prelu', bias=False)
if os.path.exists(mdlpth):
    model.load_state_dict(torch.load(mdlpth))



i = 0

init_lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr,betas=[0.9, 0.99])
#optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.0)
loss_fn = torch.nn.L1Loss()


#%%
gt_size = 64
batch_cnt = 12


tnsrss_in  = torch.ones((batch_cnt,1,gt_size,gt_size))
tnsrss_out = torch.ones((batch_cnt,1,gt_size,gt_size))

while True:
    frame_num, f_width,f_height,hq,lq = load_random_train_frame_from_vnode(train_output,train_input)
    
    lq = np.expand_dims(lq,0)
    hq = np.expand_dims(hq,0)
    
    #add extra noise
    #lq += np.random.normal(scale=0.01,size=lq.shape)

    lq = torch.Tensor(lq).cuda()
    hq = torch.Tensor(hq).cuda()
    
    fill_train_buffer(tnsrss_in,tnsrss_out,lq,hq,f_width,f_height,gt_size,batch_cnt)

    optimizer.zero_grad()
    model_outputs = model(tnsrss_in)
    loss = loss_fn(model_outputs, tnsrss_out)
    
    
    loss.backward()
    optimizer.step()
    
    save_it = 100
    if i % save_it == 0:
        mdll = model_outputs.detach().cpu().numpy()[0,0]
        hql = tnsrss_out.detach().cpu().numpy()[0,0]
        plt.subplot(141)
        plt.imshow(mdll)
        plt.title(f"model luma")
        #plt.show()
        
        plt.subplot(142)
        plt.imshow(hql)
        plt.title("hq luma")
        #plt.show()
        
        plt.subplot(143)
        plt.imshow(tnsrss_in.detach().cpu().numpy()[0,0])
        plt.title("lq luma")
        
        plt.subplot(144)
        plt.imshow((torch.abs(model_outputs - tnsrss_out)).detach().cpu().numpy()[0,0])
        plt.title("err luma")
        
        plt.show()
        #lq = frameto_numpy(train_input,a)

    if (i % save_it) == 0:
        torch.save(model.state_dict(),mdlpth)
    losnow = loss.detach().cpu().numpy()
    print(i,losnow)
    i += 1
    