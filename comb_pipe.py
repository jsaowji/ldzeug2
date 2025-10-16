import sys
from ldzeug2.comb_consts import CombConsts
from ldzeug2.stdinsrc import streaming_out,ldproject_pipe
from ldzeug2.colordecoder import *
from vstools import core, initialize_clip,vs,set_output,scale_value

from ldzeug2.utils import get_model_path

framesp,project = ldproject_pipe(sys.argv[1])
crpable = project.project["videoParameters"]["activeVideoStart"]

# I think this was for inference needs to be mod 4?
crpable_m4 = crpable % 4
crpable -= crpable_m4


import os
if "DEROT" in os.environ:
    network_path = get_model_path("color_cnn_v2_derot.onnx")
else:
    network_path = None
if "NTSC" in os.environ:
    consts=CombConsts(False)
else:
    consts=CombConsts(True)

#if "TBC_BP" in os.environ:
#    from ldzeug2.colordecoder import *
#    from ldzeug2.colordecoder import *
#    import scipy.signal as sp
#    sm = StackableManager()
#    fltrr = sp.firwin(127,[4.5,5],fs=(315/88)*4,pass_zero="bandstop")
#    f2 = fltr_to_expr(sm.add_clip(framesp), fltrr)
#    framesp = sm.eval_v(f2)



framesp = framesp.std.Crop(crpable, project.project["videoParameters"]["fieldWidth"] - project.project["videoParameters"]["activeVideoEnd"], 40, 0)
#framesp = combyc_field_model(framesp)

if "TBC_BP" in os.environ:
    framesp = comb_color_cnn_color_notch(framesp,v2=False,network_path=network_path,consts=consts)
else:
    framesp = comb_color_cnn(framesp,v2=False,network_path=network_path,consts=consts)
#import vsdenoise
#framesp  = vsdenoise.MVTools(framesp).denoise(framesp)
#fldsp = framesp.std.SeparateFields()
#import vsdenoise
#framesp  = vsdenoise.MVTools(fldsp).denoise(fldsp).std.DoubleWeave()[::2]

framesp = framesp.std.Crop(left=crpable_m4).std.AddBorders(top=2)
outnode = framesp.resize.Bicubic(format=vs.YUV444P16)

assert (outnode.width,outnode.height) == (760,488)
streaming_out(sys.stdout.buffer, outnode)