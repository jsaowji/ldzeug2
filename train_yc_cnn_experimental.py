import vapoursynth as vs
core = vs.core


core.std.LoadPlugin("/usr/lib/vapoursynth/libvsrawsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libakarin.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/bestsource.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libresize2.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libfpng.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libcolorbars.so")
core.std.LoadPlugin("/usr/lib/vapoursynth/libvslsmashsource.so")
#%%

exec(open("clips.py","rt").read())
clips: list[vs.VideoNode] = out

import json
import functools
from pathlib import Path
from vstools import vs, initialize_clip, set_output, depth, padder, Matrix, Transfer, Primaries, ColorRange,split,join
import random
import numpy as np
from matplotlib import pyplot as plt
from ldzeug2.vsnn import *
from ldzeug2.colorencoder import modulate_fields
import numpy as np
import matplotlib.pyplot as plt
#%%


og = interleave_clips(clips)
og = og.resize.Bilinear(760,486,format=vs.YUV444P16).std.SeparateFields(tff=True)

remaped = cut_to_rndm_frames(og,100)
remaped = cache_all_frames_vstools(remaped)

def asd(n,clip):
    import random
    a = random.uniform(-1, 1)
    b = random.uniform(-1, 1)

    return clip.resize.Bicubic(src_top=a,src_left=b)

p0 = 2
p1 = 3
p2 = 1

modulated_this  = modulate_fields(remaped,phaseid_at_f0=p0)
modulated_other1 = modulate_fields(core.std.FrameEval(clip=remaped,clip_src=remaped,eval=functools.partial(asd, clip=remaped)),phaseid_at_f0=p1)
modulated_other2 = modulate_fields(core.std.FrameEval(clip=remaped,clip_src=remaped,eval=functools.partial(asd, clip=remaped)),phaseid_at_f0=p2)

train_output = modulated_this.luma_out
train_input  = join([modulated_this.tbc_out, modulated_other1.tbc_out, modulated_other2.tbc_out])

#ff = lvsfunc.get_random_frames(train_input,dur=0.00000001)
#lvsfunc.export_frames(train_input,filename="/tmp/lq/%d.png",frames=ff)
#%%
from ldzeug2.compact import compact
from ldzeug2.experimentalyc import experimental
import torch
torch.set_default_device('cuda')

mdlpth = "/tmp/yc_parity_three.pth"

in_ch = 3

model = experimental(num_in_ch=in_ch, num_out_ch=1, num_feat=64, num_conv=16, upscale=1, kernel_size=3, act_type='prelu', bias=False)
model.load_state_dict(torch.load(mdlpth))





i = 0

init_lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr,betas=[0.9, 0.99])
#optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.0)
loss_fn = torch.nn.MSELoss()


#%%
gt_size = 64
batch_cnt = 12


tnsrss_in  = torch.ones((batch_cnt,in_ch,gt_size,gt_size))
tnsrss_out = torch.ones((batch_cnt,1,gt_size,gt_size))

while True:
    frame_num, f_width,f_height,hq,lq = load_random_train_frame_from_vnode(train_output,train_input)
    if in_ch == 2:
        lq = lq[0:2,:,:]
    
    
    #print(lq.shape,hq.shape)
    core.clear_cache()
    lq = np.expand_dims(lq,0)
    hq = np.expand_dims(hq,0)
    #break

    #plt.imshow(lq[0,0,:50,:50])
    #plt.show()
    #plt.imshow(lq[0,1,:50,:50])
    #plt.show()
    
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
    
    save_it = 1000
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
    