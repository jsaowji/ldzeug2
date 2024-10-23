
import json
import functools
from vstools import core, vs
from .utils import ntsc_fields_to_frames

__all__ = [
    'LDDProject',
]
#def annotate_fields(fff,ffaseat0=1):
#    import functools
#    def ano(n, f, ffaseat0):
#        fout = f.copy()
#
#        fout.props["fieldPhase"] = (ffaseat0-1 + n)%4 + 1
#        return fout
#    return fff.std.ModifyFrame(fff, functools.partial(ano, ffaseat0=ffaseat0))


class LDDProject:
    def __init__(self, tbc_path: str):
        tbcc = core.raws.Source(tbc_path, width=910, height=263, src_fmt="gray16", fpsnum=60000)
    
        project = json.load(open(tbc_path + f".json","rt"))
        fields_delta = len(tbcc) - len(project["fields"])
        if fields_delta != 0:
            print("Cropped tbcc by {} fields",fields_delta)
            tbcc = tbcc[:-fields_delta]

        def annotate_frames(n, f, project):
            fout = f.copy()
            first_field = project["fields"][n]
            fout.props["fieldPhase"] = first_field["fieldPhaseID"]
            return fout
        tbcc_flt  = tbcc.resize.Bicubic(format=vs.GRAYS,range_in=1,range=1)

        self.fields_g16 = tbcc.std.ModifyFrame(tbcc, functools.partial(annotate_frames, project=project))
        self.fields     = tbcc_flt.std.ModifyFrame(tbcc_flt, functools.partial(annotate_frames, project=project))
        self.frames = ntsc_fields_to_frames(self.fields)
        self.project = project 
