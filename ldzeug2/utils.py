from vstools import core, vs

__all__ = [
    "forcedprefetch",
    "get_model_path",
]

def forcedprefetch(a: vs.VideoNode):
    def asd(n, f, asd=[], a=a):
        for i in range(1, 10):
            a.get_frame_async(n + i, lambda c, d: 2 + 2)

        return f

    core.std.SetVideoCache(a, mode=1, fixedsize=100)
    return core.std.ModifyFrame(a, [a], selector=asd)


def get_model_path(a):
    from pathlib import Path
    import os

    return os.path.join(Path.home(), "models", a)