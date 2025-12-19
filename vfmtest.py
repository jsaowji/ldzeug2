from vstools import vs, core, set_output
from ldzeug2.lddecode import LDDProject
from ldzeug2.colordecoder import comb1d, combyc_field_model, combyc_frame_model
from vstools import depth,get_y
import os


selection = os.environ["SELECTION"]

pri = LDDProject(selection)

pri.frames = pri.frames[10000:]
kwa = { "crop": True }
c2d = comb1d(pri.frames,crop=True).std.Crop(10, 14, 6, 6)

vfmmatch = depth(get_y(c2d), 8, sample_type=vs.INTEGER).vivtc.VFM(order=True,clip2=pri.frames)


def ChangeFieldIDS(n,f):
    f_vfm = f[0]
    f_src = f[1]
    f_nxt = f[2]
    f_prv = f[3]

    fout = f[0].copy()
    vfmmatch = f_vfm.props["VFMMatch"]
    
    top = "PhaseID_A"
    bot = "PhaseID_B"
    
    #pcnbu
    if vfmmatch == 0:
        fout.props[top] = f_src.props[top]
        fout.props[bot] = f_prv.props[bot]
    elif vfmmatch == 1:
        fout.props[top] = f_src.props[top]
        fout.props[bot] = f_src.props[bot]
    elif vfmmatch == 2:
        fout.props[top] = f_src.props[top]
        fout.props[bot] = f_nxt.props[bot]
    elif vfmmatch == 3:
        fout.props[top] = f_prv.props[top]
        fout.props[bot] = f_src.props[bot]
    elif vfmmatch == 4:
        fout.props[top] = f_nxt.props[top]
        fout.props[bot] = f_src.props[bot]
    else:
        assert False

    return fout


set_output(pri.frames)
frames2 = core.std.ModifyFrame(pri.frames,[
                                           vfmmatch,
                                           pri.frames,
                                           pri.frames[1:] + pri.frames[-1],
                                           pri.frames[0] + pri.frames[:-1],
                                           
                                           ],ChangeFieldIDS)

set_output(vfmmatch)
set_output(frame_model := combyc_frame_model(frames2,crop=True))
set_output(field_model := combyc_field_model(pri.frames,crop=True))
