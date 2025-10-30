import json
import functools
from vstools import core, vs
from .ntsc_utils import ntsc_fields_to_frames

__all__ = [
    "LDDProject",
]

class LDDProject:
    def __init__(self, tbc_path: str):
        tbcc = core.raws.Source(
            tbc_path, width=910, height=263, src_fmt="gray16", fpsnum=60000
        )

        project = json.load(open(tbc_path + ".json", "rt"))
        fields_delta = len(tbcc) - len(project["fields"])
        
        if fields_delta != 0:
            print(".tbc file and fields in json dont match: cropped tbc by {} fields", fields_delta)
            tbcc = tbcc[:-fields_delta]

        def annotate_frames(n, f, project):
            fout = f.copy()
            first_field = project["fields"][n]
            fout.props["fieldPhase"] = first_field["fieldPhaseID"]
            return fout

        tbcc_flt = tbcc.resize.Bicubic(format=vs.GRAYS, range_in=1, range=1)

        self.fields_g16 = tbcc.std.ModifyFrame(
            tbcc, functools.partial(annotate_frames, project=project)
        )
        self.fields = tbcc_flt.std.ModifyFrame(
            tbcc_flt, functools.partial(annotate_frames, project=project)
        )
        self.frames = ntsc_fields_to_frames(self.fields)
        self.frames_g16 = ntsc_fields_to_frames(self.fields_g16)

        self.project = project
