#%%
import vapoursynth as vs
core = vs.core


core.std.LoadPlugin("/usr/lib/vapoursynth/libvsrawsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libakarin.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/bestsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libresize2.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libaddgrain.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libcolorbars.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libvslsmashsource.so")
#%%
import json
import functools
from pathlib import Path
from vstools import remap_frames, vs, initialize_clip, set_output, depth, padder, Matrix, Transfer, Primaries, ColorRange,split,join
import random
import numpy as np
from matplotlib import pyplot as plt
from ldzeug2.vsnn import fill_train_buffer,cut_to_rndm_frames,cache_all_frames, get_rndm_frames,interleave_clips,load_random_train_frame_from_vnode, load_random_train_frame_from_vnode_idx_given
from ldzeug2.colorencoder import modulate_fields
import numpy as np
import matplotlib.pyplot as plt
import vskernels as vke

from ldzeug2.stackable import *
def add_dropouts(frmz):
    sm = StackableManager()
    s = sm.add_clip(frmz)
    grnd = sm.add_clip(gg:=frmz.grain.Add(var=100000,hcorr=0.9))
    x = 10
    isy = s * 0.0
    for _ in range(80):
        x = random.randint(2,6)
        posi = random.randint(0,frmz.width-1)
        posi2 = random.randint(0,frmz.height-1)
        isy += ((oX >= posi) & (oX <= (posi+x))) & (oY == posi2)

    frmz = sm.eval_v(isy.iftrue(grnd,s))
    return frmz

def lowypassy(a:vs.VideoNode,ffr=5.5e6):
    import numpy as np
    def ffn(n,f,ffr=ffr):
        npf = np.array(f[0])
        import scipy.signal as sp
        b,a = sp.butter(3,ffr,fs=3.5e6*4)

        npf = sp.filtfilt(b,a,npf)

        
        f2 = f.copy()
        np.copyto(np.asarray(f2[0]),npf)
        return f2

    return a.std.ModifyFrame(a,ffn)


exec(open("clips.py","rt").read())
clips: list[vs.VideoNode] = out
og = interleave_clips(clips)

on_fields = True

kernel_in  = vke.Bicubic()
#kernel_out = vke.Gaussian(sigma=0.5)
kernel_out = kernel_in
ccnt = 180
if on_fields:
    og_in  = kernel_in.scale(og,760,486,format=vs.YUV444P16).std.SeparateFields(tff=True)
    og_out = kernel_out.scale(og,760,486,format=vs.YUV444P16).std.SeparateFields(tff=True)
    frms = get_rndm_frames(og)
    
    #from vstools import remap_frames
    og_in = remap_frames(og_in,frms[:ccnt])
    og_out = remap_frames(og_out,frms[:ccnt])
else:
    assert False
    og = kernel.scale(og,760,486,format=vs.YUV444P16)
    remaped = cut_to_rndm_frames(og,ccnt).std.SeparateFields(tff=True)

remaped_in  = core.std.Interleave([og_in,og_in])
remaped_out = core.std.Interleave([og_out,og_out])

modulated_in  = modulate_fields(remaped_in)
modulated_out = modulate_fields(remaped_out)

train_input  = join([lowypassy(add_dropouts(modulated_in.tbc_out)), modulated_in.i_carier,modulated_in.q_carier])
train_output = join([modulated_out.luma_out,modulated_out.i_lp,modulated_out.q_lp])

if on_fields:
    pass
else:
    train_output = train_output.std.DoubleWeave(tff=True)[::2]
    train_input  = train_input.std.DoubleWeave(tff=True)[::2]

print("caching")
cache_all_frames(train_input)
cache_all_frames(train_output)
#%%
import pyqtgraph as pg
%gui qt5
pg.image(np.array([
    np.array(train_output.get_frame(0)[0]).transpose(),
    np.array(train_input.get_frame(0)[0]).transpose(),
]))
#%%
import pyqtgraph as pg
%gui qt5
pg.image(np.array([
    np.array(train_output.get_frame(0)[1]).transpose(),
    np.array(train_input.get_frame(0)[1]).transpose(),
]))
#%%
import torch
import os

torch.set_default_device('cuda')
mdlpth = "/tmp/rr48.pth"

from ldzeug2.colorcnn_trch import FullModel2
model = FullModel2(num_feat=64,num_conv=16)
if os.path.exists(mdlpth):
    model.load_state_dict(torch.load(mdlpth))


i = 0

init_lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr,betas=[0.9, 0.99])
#optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.0)

loss_fn = torch.nn.MSELoss()


#%%
gt_size = 64
batch_cnt = 12
save_it = 1000

tnsrss_in  = torch.ones((batch_cnt,3,gt_size,gt_size))
tnsrss_out = torch.ones((batch_cnt,3,gt_size,gt_size))

while True:
    frame_num, f_width,f_height,hq,lq = load_random_train_frame_from_vnode_idx_given((i % len(train_output) // 2) * 2+0,train_output,train_input)
    lq = torch.Tensor(np.expand_dims(lq,0)).cuda()
    hq = torch.Tensor(np.expand_dims(hq,0)).cuda()
    
    fill_train_buffer(tnsrss_in,tnsrss_out,lq,hq,f_width,f_height,gt_size,batch_cnt//2,0)
    frame_num, f_width,f_height,hq,lq = load_random_train_frame_from_vnode_idx_given((i % len(train_output) // 2) * 2+1,train_output,train_input)
    lq = torch.Tensor(np.expand_dims(lq,0)).cuda()
    hq = torch.Tensor(np.expand_dims(hq,0)).cuda()
    

    fill_train_buffer(tnsrss_in,tnsrss_out,lq,hq,f_width,f_height,gt_size,batch_cnt//2,batch_cnt//2)

    optimizer.zero_grad()
    model_outputs = model(tnsrss_in)
    loss = loss_fn(model_outputs, tnsrss_out)
    
    loss.backward()
    optimizer.step()
    

    if (i % save_it) == 0:
        torch.save(model.state_dict(),mdlpth)
    losnow = loss.detach().cpu().numpy()
    print(i,losnow)

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

    i += 1
    
# %%
