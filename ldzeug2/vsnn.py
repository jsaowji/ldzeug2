from vstools import core,vs
import random
import numpy as np

def interleave_clips(a: list[vs.VideoNode]):
    smallest_len = 2**64
    for b in a:
        smallest_len = min(smallest_len,len(b))
    return core.std.Interleave([c[:smallest_len] for c in a] )

def cut_to_rndm_frames(og: vs.VideoNode,frmcn) -> vs.VideoNode:
    frms = list(range(len(og)))
    random.shuffle(frms)

    from vstools import remap_frames
    return remap_frames(og,frms[:frmcn])

def cache_all_frames(a: vs.VideoNode):
    a.std.SetVideoCache(1,2*len(a)+50,2*len(a)+50)
    for i,a in enumerate(a.frames()):
        print(i,len(a))
def cache_all_frames_vstools(inp: vs.VideoNode) -> vs.VideoNode:
    from vstools import cache_clip

    a = cache_clip(inp,cache_size=len(inp))
    for i,af in enumerate(a.frames()):
        print(i,len(a))
    return a
        
def get_rndm_crop(width,height,crpp):
    xof = random.randint(0,width)
    yof = random.randint(0,height)
    
    #xof = xof - max(0,xof-width-crpp)
    #yof = yof - max(0,yof-height-crpp)
    if xof + crpp > width:
        xof = width - crpp
    if yof + crpp > height:
        yof = height - crpp
    
    yof -= yof % 2
    xof -= xof % 4
    
    return xof,yof


def frameto_numpy(xx:vs.VideoNode,a:int):
    hqf = xx.get_frame(a)
    hq = np.ones( (len(hqf), xx.height, xx.width) )
    
    for abb in range(len(hqf)):
        hq[abb,:,:] = np.asarray(hqf[abb])
    return hq

def load_random_train_frame_from_vnode(train_output: vs.VideoNode, train_input: vs.VideoNode):
    a = random.randint(0,len(train_output)-1)
    
    hq = frameto_numpy(train_output,a)
    lq = frameto_numpy(train_input,a)
    
    f_height = train_output.height
    f_width = train_output.width
    return a,f_width,f_height,hq,lq

def fill_train_buffer(tnsrss_in,tnsrss_out,lq,hq,f_width,f_height,gt_size,batch_cnt):
    for a in range(batch_cnt):
        xof,yof = get_rndm_crop(f_width, f_height, gt_size)
        inputs_cr  = lq          [:,:,  yof:(yof+gt_size), xof:(xof+gt_size)]
        outputs_cr = hq          [:,:,  yof:(yof+gt_size), xof:(xof+gt_size)]
        tnsrss_in [a, : , : , :] = inputs_cr
        tnsrss_out[a, : , : , :] = outputs_cr

def sanity_check_model():
    pass
    #x = torch.empty((1,1,263,910))
    #assert x.shape == model(x).shape
    #
    #
    #
    #for _ in range(100):
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



def torch_model_frame_eval(node: vs.VideoNode, output: vs.VideoNode,model) -> vs.VideoNode:
    def apply_model22(n,f,model):
        import numpy as np
        import torch
        from torch import Tensor
        frm0 = Tensor(np.array(f[0])).unsqueeze(0)
        out = model(frm0)
        fout = f[1].copy()
        for i in range(len(fout)):
            np.copyto(np.asarray(fout[i]),out.detach().cpu().numpy()[0,i])
        return fout
    
    return core.std.ModifyFrame(output,[node,output],apply_model22)

