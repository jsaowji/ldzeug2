import numpy as np
from vstools import core,vs,cache_clip
from .lddecode import LDDProject

__all__ = [
    "streaming_in",
    "streaming_out",
    "ldproject_pipe",
]

#Some ffmpeg commands:
# | ffmpeg -video_size 910x263 -pixel_format y16 -r 60000/1001 -f rawvideo -i - -strict -1 -f yuv4mpegpipe - | mpv --pause=no -                              
# | ffmpeg -video_size 760x486 -pixel_format yuv422p -r 30000/1001 -f rawvideo -i - -strict -1 -f yuv4mpegpipe - | mpv --pause=no -                              
# | ffmpeg -video_size 910x526 -pixel_format yuv444p -r 30000/1001 -f rawvideo -i - -strict -1 -f yuv4mpegpipe - | mpv --pause=no -                              

def ldproject_pipe(ld_decode_project: str) -> tuple[vs.VideoNode,LDDProject]:
    import sys
    project = LDDProject(ld_decode_project)

    #orig_tbc,project = ldproject_frames(ld_decode_project,addjson,extension)

    tbc = streaming_in(sys.stdin.buffer)
    tbc = tbc.resize.Bicubic(format=vs.GRAYS,range_in=1,range=1)
    tbc = core.std.DoubleWeave(tbc,tff=True)[::2]
    tbc = core.std.CopyFrameProps(tbc,project.frames,props=["PhaseID_A","PhaseID_B","fieldPhaseID"])
    return tbc, project

def streaming_in(file_handle, ww=910, hh=263):
    bytespersample = 2
    def get_frm_call(n,f,ftched=[-1],file_handle=file_handle):
        #print("fetch internal", n)
        assert n == ftched[0]+1
        ftched[0] += 1

        read_size = ww*hh * bytespersample
        bb = file_handle.read(read_size)

        nf = f.copy()
        nf.get_write_ptr(0)

        if len(bb) == read_size:
            raws = np.frombuffer(bb,dtype=np.uint16)
            frm = raws.reshape((hh,ww,))
            np.copyto(np.asarray(nf[0]),frm)
            nf.props["EOF"] = 0
        else:
            nf.props["EOF"] = 1

        return nf
    
    blk_clp = core.std.BlankClip(format=vs.GRAY16,width=ww,height=hh,length=200000000)
    clp = blk_clp

    clp = core.std.ModifyFrame(clp, clp, get_frm_call)
    clp = cache_clip(clp,40)


    def reorderre(n,f,ftched=[-1],inna=clp):
       # print("reorderer ftch ",n)
        
        if n < ftched[0]+1:
            # hope and pray its cached layer down
            return inna.get_frame(n)
        assert n >= ftched[0]+1

        #fetch from src in right order
        while n != ftched[0] + 1:
            inna.get_frame(ftched[0]+1)
            ftched[0] += 1
        return inna.get_frame(n)

    clp = core.std.ModifyFrame(clp, blk_clp, reorderre)
    clp = cache_clip(clp,40)
    return clp

import sys

def streaming_out(file_handle, v:vs.VideoNode):
    for f in v.frames():
        #sys.stderr.write("wrote frame")
        plane_cnt = len(f)
        if f.props["EOF"] == 1:
            sys.stderr.write("EOF")
            break
        for p in range(plane_cnt):
            file_handle.write(bytes(np.array(f[p])))
