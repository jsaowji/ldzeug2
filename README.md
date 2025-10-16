# ldzeug2
Tools centered around vapoursynth for dealing with composite video sourced from NTSC(-J) laserdiscs.
Contains vapoursynth pcm audio source, ld-decode comb filter ported over to vs, tools for using cnn based models for y/c seperation, color decoding and more.
Provided without warrenty or guarantee for correctness there are some rough edges. I consider this repo complete in the sense that if I visit this again it will get a full rewrite. Fixes still welcome though.
Pretrained models can be found in Releases and should be put wherever ldzeug2/utils.py/get_model_path evaluates to on your os.

## ld-decode's comb filter ported to vapoursynth + extra
```py
from ldzeug2.lddecode import LDDProject
from ldzeug2.colordecoder import *
from vstools import *
from vsmlrt import BackendV2

pri = LDDProject("example.tbc")

kwa = { "crop": True }

set_output(comb1d(pri.frames,**kwa),"1d")
set_output(comb2d(pri.frames,**kwa),"2d")
set_output(comb3d(pri.frames,**kwa),"3d")

# model (per field) about as good as 3d
# + no ghosting on fast motion
# - less detail on stationary
set_output(combyc_field_model(pri.frames,backend=BackendV2.TRT(),**kwa),"cnn per field")
set_output(combyc_frame_model(pri.frames,backend=BackendV2.TRT(),**kwa),"cnn per frame")

# good aswell
set_output(comb_color_cnn(pri.frames,v2=False,crop=True),"color cnn v1")
set_output(comb_color_cnn(pri.frames,v2=True,crop=True), "color cnn v2")
```

## vapoursynth audio source for ld-decoded pcm files
```py
core.ldpcmaudio.Source("<...>")
```

## Play color decoded .tbc directly with mpv + audio (vswatch)
```
#digital audio is picked up if placed at "out/efm.raw" in folder of tbc
python -m vswatch play.py example.tbc
``` 

## Encoding .tbc directly from vs
```
# comb_pipe.py always reads video data from stdin example.tbc only for loading json info
ld-dropout-correct --output-json /tmp/a.json example.tbc - | python comb_pipe.py example.tbc | ffmpeg -video_size 760x486 -pixel_format yuv444p16 -r 30000/1001 -f rawvideo -i /dev/stdin  <.....>
```

# Requirements (not all are required for every function)
- vstools from JET Suite (py)
- vsmlrt (py + vs)
- fillborders (vs)
- sympy (py)
- scipy (py)
- numpy (py)
- matplotlib (py)
- github.com/jsaowji/vswatch (py)