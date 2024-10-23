from vstools import vs, core, split,join
import sys
import functools
import math
from ldzeug2.comb_consts import CombConsts,COLOR_CARIER_FREQ_FLT
from .stackable import *
from dataclasses import dataclass

__all__ = [
    'FieldModulatedOutput',
    'modulate_fields',

    'annon_phase_frames_to_fields',
]

#?? how was this designed
iq_lowpass = [0.0021, 0.0191, 0.0903, 0.2308, 0.3153, 0.2308, 0.0903, 0.0191, 0.0021]

import numpy as np



@dataclass
class FieldModulatedOutput:
    tbc_out: vs.VideoNode
    luma_out: vs.VideoNode
    chroma_out: vs.VideoNode
    i_carier: vs.VideoNode
    q_carier: vs.VideoNode
    i_hp: vs.VideoNode
    q_hp: vs.VideoNode

def annon_phase_frames_to_fields(tbcc2_o_g):
    def modifyrm(n,f):
        f2 = f.copy()
        f2.props["pid"] = [f2.props["PhaseID_B"],f2.props["PhaseID_A"]][f2.props["_Field"]]
        return f2
    fields = tbcc2_o_g.std.SeparateFields(tff=True)
    bb = core.std.ModifyFrame(fields,fields,modifyrm)
    return bb

def do_splitin(tbcc2_o_g):
    #tbcc2_o_g = tbcc2_o_g.std.Crop(left=2)
    bb = annon_phase_frames_to_fields(tbcc2_o_g)
    def modifyrm2(n,f,ofst):
        import numpy as np
        f2 = f[1].copy()
        f2.props = f[0].props
        f2.get_write_ptr(0)

        inp = np.asarray(f[0])

        assert (f[0].width % 4) == 0
        out = np.empty((1,f[0].height,f[0].width//4))

        mping = [2,3,0,1]

        if f[0].props["pid"] in [2,3]:
            ofst = [2, 3, 0,1][ofst]
        out[:,  :, :]    = inp[:,:,(ofst)::4]
        out[:, 1::2, :]  = inp[:,1::2,mping[ofst]::4]

        np.copyto(np.asarray(f2[0]),out)
        return f2
    prr = core.std.BlankClip(bb,width=bb.width // 4)
    m = [ core.std.ModifyFrame(prr,[bb,prr],functools.partial(modifyrm2,ofst=a)).resize.Point(width=bb.width) for a in range(0,4)]
    return [bb] + m


def iq_from_uv(u1: Stackable,
               v1:Stackable,
               chromaPhase = 0):
    chromaGain = 1

    theta = ((33 + chromaPhase) * math.pi) / 180
    bp1 = math.sin(theta) * chromaGain
    bq1 = math.cos(theta) * chromaGain


    #TOOD: hardcode this to remove sympy req
#    u = i * -bp  + q * bq
#    v = i * bq   + q * bp
    import sympy
    u,v,i,q,bp,bq = sympy.symbols("u v i q bp bq")
    m2 = sympy.Matrix([[-bp, bq], [bq, bp]]).inv()
    m2 = m2 * sympy.Matrix([u,v])
    m2 = m2.subs({bp:bp1,bq:bq1})

    #print(m2[0].subs({u:1,v:0}))

    i = u1 * float(m2[0].subs({u:1,v:0})) +  v1 * float(m2[0].subs({u:0,v:1}))
    q = u1 * float(m2[1].subs({u:1,v:0})) +  v1 * float(m2[1].subs({u:0,v:1}))

    return i,q

def int_to_floatsm(sm,a:vs.VideoNode):
    # convert to float
    assert a.format.id == vs.YUV444P16
    a = split(a)
    yo = sm.add_clip(a[0].std.Expr("x 256 / 256 /",format=vs.GRAYS))
    cb = sm.add_clip(a[1].std.Expr("x 256 / 256 /",format=vs.GRAYS))
    cr = sm.add_clip(a[2].std.Expr("x 256 / 256 /",format=vs.GRAYS))
    return yo,cb,cr

def ycbcr_to_yuv(yo:Stackable,cb:Stackable,cr:Stackable,consts:CombConsts):
    y = ((yo  - consts.Y_ZERO) / consts.yScale) + consts.yOffset
    u = (cb - consts.C_ZERO) / consts.cbScale
    v = (cr - consts.C_ZERO) / consts.crScale
    return y,u,v


def modulate_fields(a: vs.VideoNode, phaseid_at_f0: int = 1, consts=CombConsts(True), x_offset:int = 0) -> FieldModulatedOutput:
    assert phaseid_at_f0 in [1,2,3,4]
    sm = StackableManager()

    yo,cb,cr = int_to_floatsm(sm,a)
    y,u,v = ycbcr_to_yuv(yo,cb,cr,consts)
    i_hp, q_hp = iq_from_uv(u,v)
    fsc =   (315/88) * 1_000_000

    #color carriere
    cc = ( ((oX + x_offset) / (4.0*fsc)) * 2 * np.pi * fsc  + ( (-90) * (np.pi / 180)) )

    phase_now = (((oN+phaseid_at_f0-1) % 4)+1)
    swtch = ( (oY % 2) == 0).iftrue(1.0,-1.0) * phase_now.switch([
        (1,1),
        (2,-1),
        (3,-1),
        (4,1), 
    ])

    i = fltr_to_expr(i_hp,iq_lowpass)
    q = fltr_to_expr(q_hp,iq_lowpass) 
    
    mod1 = cc.cos() * swtch
    mod2 = cc.sin() * swtch
    #set_output(sm.eval_v(cc.cos() * swtch))
    chrm = (i * mod1 + q * mod2)
    out = y + chrm
    mm =  sm.eval_v(out)

    def tag_fieldPhase(n,f):
        f2 = f.copy()
        f2.props["fieldPhase"] =  (((n+phaseid_at_f0-1) % 4)+1)

        return f2
        
    return FieldModulatedOutput(core.std.ModifyFrame(mm,mm,tag_fieldPhase),
                                sm.eval_v(y),
                                sm.eval_v(chrm),
                                sm.eval_v(mod1),
                                sm.eval_v(mod2),
                                sm.eval_v(i_hp),
                                sm.eval_v(q_hp))