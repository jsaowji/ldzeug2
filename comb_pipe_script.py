import sys
from ldzeug2.stdinsrc import streaming_out,ldproject_pipe
from ldzeug2.colordecoder import combyc_field_model,comb_color_cnn
from vstools import core, initialize_clip,vs,set_output,scale_value

video_in,project = ldproject_pipe(sys.argv[1])
scrptpth = sys.argv[2]

eval(open(scrptpth,"rt").read())
outnode = vs.get_output(0).clip

assert (outnode.width,outnode.height) == (760,486)
streaming_out(sys.stdout.buffer, outnode)