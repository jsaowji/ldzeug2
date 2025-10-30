from vstools import core, vs
import functools

__all__ = [
    "ntsc_fields_to_frames",
    "ntsc_frames_to_fields",
    "ntsc_fields_tag_phases",
    "ntsc_doubleweave",
]


def ntsc_fields_tag_phases(fields_in: vs.VideoNode, phaseid_at_f0: int) -> vs.VideoNode:
    def tag_fieldPhase(n, f, phaseid_at_f0=phaseid_at_f0):
        f2 = f.copy()
        f2.props["fieldPhase"] = ((n + phaseid_at_f0 - 1) % 4) + 1

        return f2

    return core.std.ModifyFrame(fields_in, fields_in, tag_fieldPhase)


def ntsc_fields_to_frames(fields_in: vs.VideoNode) -> vs.VideoNode:
    def annotate_frames(n, f, fields_in=fields_in):
        fout = f.copy()
        first_field = fields_in.get_frame(n * 2)
        second_field = fields_in.get_frame(n * 2 + 1)
        fout.props["PhaseID_A"] = first_field.props["fieldPhase"]
        fout.props["PhaseID_B"] = second_field.props["fieldPhase"]
        return fout

    tbcc2 = core.std.DoubleWeave(fields_in, tff=True)[::2]
    tbcc2 = tbcc2.std.ModifyFrame(tbcc2, functools.partial(annotate_frames))

    return tbcc2


def ntsc_frames_to_fields(frames_in: vs.VideoNode) -> vs.VideoNode:
    def annotate_fields(n, f, frames_in=frames_in):
        fout = f.copy()
        #TODO: illegal
        first_field = frames_in.get_frame(n // 2)
        fout.props["fieldPhase"] = [
            first_field.props["PhaseID_B"],
            first_field.props["PhaseID_A"],
        ][f.props["_Field"]]
        return fout

    tbcc3 = core.std.SeparateFields(frames_in, tff=True)
    tbcc3 = tbcc3.std.ModifyFrame(tbcc3, functools.partial(annotate_fields))

    return tbcc3


def ntsc_doubleweave(fields_in: vs.VideoNode) -> vs.VideoNode:
    def annotate_fields(n, f, fields_in=fields_in):
        fout = f.copy()
        if n % 2 == 0:
            first_field = fields_in.get_frame(n)
            second_field = fields_in.get_frame(n + 1)

            fout.props["PhaseID_A"] = first_field.props["fieldPhase"]
            fout.props["PhaseID_B"] = second_field.props["fieldPhase"]
        else:
            first_field = fields_in.get_frame(n)
            second_field = fields_in.get_frame(n + 1)

            fout.props["PhaseID_A"] = second_field.props["fieldPhase"]
            fout.props["PhaseID_B"] = first_field.props["fieldPhase"]

        return fout

    tbcc2 = core.std.DoubleWeave(fields_in, tff=True)
    tbcc2 = tbcc2.std.ModifyFrame(tbcc2, functools.partial(annotate_fields))

    return tbcc2