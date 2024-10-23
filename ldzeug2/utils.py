from vstools import core,vs
import functools

__all__ = [
    "ntsc_fields_to_frames",
    "forcedprefetch",
    'get_model_path'
]

def ntsc_fields_to_frames(fields_in: vs.VideoNode) -> vs.VideoNode:
    def annotate_frames(n, f, fields_in=fields_in):
        fout = f.copy()
        first_field = fields_in.get_frame(n * 2)
        second_field = fields_in.get_frame(n * 2 + 1)
        fout.props["PhaseID_A"] = first_field.props["fieldPhase"]
        fout.props["PhaseID_B"] = second_field.props["fieldPhase"]
        return fout

    tbcc2 = core.std.DoubleWeave(fields_in,tff=True)[::2]
    tbcc2 = tbcc2.std.ModifyFrame(tbcc2, functools.partial(annotate_frames))

    
    return tbcc2

def forcedprefetch(a:vs.VideoNode):
#    return a
    def asd(n,f,asd=[],a=a):
        for i in range(1,10):
            a.get_frame_async(n+i, lambda c,d: 2+2)

        return f
    core.std.SetVideoCache(a,mode=1,fixedsize=100)
    return core.std.ModifyFrame(a,[a],selector=asd)


def get_model_path(a):
    from pathlib import Path
    import os
    return os.path.join(Path.home(),"models", a)

def plot_filter(b):
    import numpy as np
    from scipy import signal
    from matplotlib import pyplot as plt
    w, h = signal.freqz(b,fs=4)
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    #ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.plot(w, abs(h), 'b')

    ax1.set_ylabel('Amplitude []', color='b')
    ax1.set_xlabel('Frequency []')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    plt.show()
