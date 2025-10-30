from numpy import random
from vstools import vs

from .stackable import StackableManager, oX, oY
import scipy.signal as sp
import numpy as np


__all__ = [
    "add_dropouts",
    "hor_butter_lowpass"
]

def add_dropouts(frmz: vs.VideoNode) -> vs.VideoNode:
    sm = StackableManager()
    s = sm.add_clip(frmz)
    grnd = sm.add_clip(frmz.grain.Add(var=100000,hcorr=0.9))
    x = 10
    isy = s * 0.0
    for _ in range(80):
        x = random.randint(2,6)
        posi = random.randint(0,frmz.width-1)
        posi2 = random.randint(0,frmz.height-1)
        isy += ((oX >= posi) & (oX <= (posi+x))) & (oY == posi2)

    frmz = sm.eval_v(isy.iftrue(grnd,s))
    return frmz

def hor_butter_lowpass(a: vs.VideoNode, ffr=5.5e6):

    def ffn(n,f,ffr=ffr):
        npf = np.array(f[0])
        b,a = sp.butter(3,ffr,fs=3.5e6*4)

        npf = sp.filtfilt(b,a,npf)

        f2 = f.copy()
        np.copyto(np.asarray(f2[0]),npf)
        return f2

    return a.std.ModifyFrame(a,ffn)
