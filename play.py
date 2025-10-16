from vstools import set_output, core, vs, Sar, Dar
from ldzeug2.colordecoder import *
from ldzeug2.lddecode import *
from ldzeug2.utils import *
import os

selection: str = globals()["selection"]

project = LDDProject(selection)
audio_pth = selection[:-4] + ".pcm"
has_audio = os.path.exists(audio_pth)
if has_audio:
    aa = core.ldpcmaudio.Source(audio_pth).std.AssumeSampleRate(samplerate=project.project["pcmAudioParameters"]["sampleRate"])


import os

digiaudio = os.path.join(os.path.dirname(selection),"out/efm.raw")
aa2 = None
if os.path.exists(digiaudio):
    aa2 = core.ldpcmaudio.Source(digiaudio)

#nd = comb2d(project.frames, crop=True).resize.Bicubic(format=vs.YUV422P8,matrix_in_s="170m")

nd = comb_color_cnn(project.frames,v2=False,crop=True,consts=CombConsts("NTSC" not in os.environ)).resize.Bicubic(format=vs.YUV422P8,matrix_in_s="170m")
#nd = comb3d(project.frames, crop=True).std.AddBorders(top=2).resize.Bicubic(format=vs.YUV420P8,matrix_in_s="170m")

nd = Sar.from_ar(height=nd.height,active_area=nd.width,dar=Dar(4,3)).apply(nd)

#set_output(forcedprefetch(nd))
#set_output(aa)
#if aa2 is not None:
#    set_output(aa2)

forcedprefetch(nd).set_output(0)
if has_audio:
    aa.set_output(1)

if aa2 is not None:
    aa2.set_output(2)

