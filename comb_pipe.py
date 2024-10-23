import sys
from ldzeug2.stdinsrc import streaming_out,ldproject_pipe
from ldzeug2.colordecoder import combyc_field_model,comb_color_cnn
from vstools import core, initialize_clip,vs,set_output,scale_value

framesp,project = ldproject_pipe(sys.argv[1])
crpable = project.project["videoParameters"]["activeVideoStart"]

# I think this was for inference needs to be mod 4?
crpable_m4 = crpable % 4
crpable -= crpable_m4



framesp = framesp.std.Crop(crpable, project.project["videoParameters"]["fieldWidth"] - project.project["videoParameters"]["activeVideoEnd"], 40, 0)
#framesp = combyc_field_model(framesp)
framesp = comb_color_cnn(framesp,v2=False)

#import vsdenoise
#framesp  = vsdenoise.MVTools(framesp).denoise(framesp)
#fldsp = framesp.std.SeparateFields()
#import vsdenoise
#framesp  = vsdenoise.MVTools(fldsp).denoise(fldsp).std.DoubleWeave()[::2]

framesp = framesp.std.Crop(left=crpable_m4)
outnode = framesp.resize.Bicubic(format=vs.YUV444P16)

assert (outnode.width,outnode.height) == (760,486)
streaming_out(sys.stdout.buffer, outnode)