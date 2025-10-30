from __future__ import annotations
from vstools import core, vs, split, join
from dataclasses import dataclass
from typing import Optional
from .stackable import StackableManager, Stackable, oY, oX, fltr_to_expr
from .utils import get_model_path
from .ntsc_utils import ntsc_frames_to_fields

from .comb_consts import COLOR_CARIER_FREQ_FLT, CombConsts
import math

__all__ = [
    "comb1d",
    "comb2d",
    "comb3d",
    "combyc_field_model",
    "combyc_frame_model",
    "comb_color_cnn",
    "uv_from_iq",
    "to_yuv",
    "comb_split_already",
]


def color_notch_bp(
    framesp: vs.VideoNode, f1: float = 4.5, f2: float = 5.5
) -> vs.VideoNode:
    import scipy.signal as sp

    sm = StackableManager()
    fltrr = sp.firwin(127, [f1, f2], fs=(315 / 88) * 4, pass_zero="bandstop")
    fefe = fltr_to_expr(sm.add_clip(framesp), fltrr)
    return sm.eval_v(fefe)


def color_notch_lp(framesp: vs.VideoNode, freq: float = 4.5) -> vs.VideoNode:
    import scipy.signal as sp

    sm = StackableManager()
    fltrr = sp.firwin(127, freq, fs=(315 / 88) * 4, pass_zero="lowpass")
    f2 = fltr_to_expr(sm.add_clip(framesp), fltrr)
    return sm.eval_v(f2)


# for 4fsc
c_colorlp_b = [
    2.236562025869846e-03,
    9.679572273064329e-03,
    6.100849475810623e-03,
    -2.082153208645807e-02,
    -4.872917723725065e-02,
    -1.535300561979003e-02,
    1.137084573789944e-01,
    2.775133160099456e-01,
    3.533666167131518e-01,
    2.775133160099456e-01,
    1.137084573789944e-01,
    -1.535300561979003e-02,
    -4.872917723725065e-02,
    -2.082153208645807e-02,
    6.100849475810623e-03,
    9.679572273064329e-03,
    2.236562025869846e-03,
]


def phase_flip_props(clip: Stackable) -> Stackable:
    oY4 = oY % 4
    phase_id_table = [
        (1, (oY4 == 0).iftrue(-1, 1)),
        (2, (oY4 == 1).iftrue(1, -1)),
        (3, (oY4 == 0).iftrue(1, -1)),
        (4, (oY4 == 1).iftrue(-1, 1)),
    ]
    # who big hack
    return (oY % 2).switch(
        [
            (0, clip["PhaseID_A"].switch(phase_id_table)),
            (1, clip["PhaseID_B"].switch(phase_id_table)),
        ]
    )


iq_phase = oX % 4


#  0  1  2  3
# +Q -I -Q +I


def split_i(x: Stackable) -> Stackable:
    return iq_phase.switch([(0, x[-1, 0]), (1, -x), (2, -x[-1, 0]), (3, x)])
    # 0 1  2 3
    # | -X | X


# clr chroma with same phased
def split_q(x: Stackable):
    return iq_phase.switch([(0, x), (1, x[-1, 0]), (2, -x), (3, -x[-1, 0])])
    # 0 1 2  3
    # X | -X |


def split_iq(x: Stackable) -> tuple[Stackable, Stackable]:
    i = split_i(x)
    q = split_q(x)
    return i, q


def uv_from_iq(i: Stackable, q: Stackable, chromaPhase=0.0, chroma_gain=1.0):
    import math

    # chromaPhase = 0#-90
    chromaGain = chroma_gain

    theta = ((33 + chromaPhase) * math.pi) / 180
    bp = math.sin(theta) * chromaGain
    bq = math.cos(theta) * chromaGain

    u = i * -bp + q * bq
    v = i * bq + q * bp
    return u, v


def adjust_y(
    signal: Stackable, i: Stackable, q: Stackable, phase_fliped: Stackable
) -> Stackable:
    return signal - (
        iq_phase.switch([(0, -q), (1, i), (2, q), (3, -i)]) * phase_fliped * -1
    )


def to_rgb(
    y: Stackable, u: Stackable, v: Stackable, consts: CombConsts
) -> tuple[Stackable, Stackable, Stackable]:
    yScale = 65535.0 / consts.yRange
    uvScale = 65535.0 / consts.uvRange

    rY = (y - consts.yOffset) * yScale
    rU = u * uvScale
    rV = v * uvScale

    r = rY + (rV * 1.139883)
    g = rY + (rU * -0.394642) + (rV * -0.580622)
    b = rY + (rU * 2.032062)
    return r, g, b


def to_yuv(
    y: Stackable, u: Stackable, v: Stackable, consts: CombConsts
) -> tuple[Stackable, Stackable, Stackable]:
    y = (y - consts.yOffset) * consts.yScale + consts.Y_ZERO
    cb = (u * consts.cbScale) + consts.C_ZERO
    cr = (v * consts.crScale) + consts.C_ZERO

    y = y.clamp(consts.Y_MIN, consts.Y_MAX)
    cb = cb.clamp(consts.C_MIN, consts.C_MAX)
    cr = cr.clamp(consts.C_MIN, consts.C_MAX)

    return y, cb, cr


def comb_split_already(
    vid_v: vs.VideoNode,
    chrma: vs.VideoNode,
    consts: CombConsts = CombConsts(True),
    lma_override: vs.VideoNode = None,
    chroma_phase: float = 0,
    chroma_gain: float = 1.0,
    luma_gain: float = 1.0,
    crop: bool = False,
    color_bp=True,
    color_notch=False,
    color_low=False,
    do_rgb=False,
    keep_iq=False,
    iq_cariers=None,
):
    if vid_v.format.id == vs.GRAY16:
        vid_v = vid_v.resize.Bicubic(format=vs.GRAYS)
    if chrma.format.id == vs.GRAY16:
        chrma = chrma.resize.Bicubic(format=vs.GRAYS)
    assert chrma.format.id == vid_v.format.id == vs.GRAYS  # lma.format.id ==

    sm = StackableManager()
    vid_x = sm.add_clip(vid_v)
    chrma_x = sm.add_clip(chrma)

    pkpk1 = phase_flip_props(vid_x)

    # wokraround some bug in the stackable system
    # where relative acces to Stackable doesnt work properly but to raw clip it does
    pkpk = sm.eval_x(pkpk1)

    i, q = split_iq(chrma_x * pkpk)
    if iq_cariers is not None:
        mod1, mod2 = [sm.add_clip(a) for a in iq_cariers]
        i, q = chrma_x * mod1, chrma_x * mod2

    i, q = sm.eval_x([i, q])

    y1 = adjust_y(vid_x, i, q, pkpk)

    if lma_override is not None:
        y1 = sm.add_clip(lma_override)
    y1 = y1 * luma_gain

    if color_low:
        if isinstance(color_low, bool):
            color_low = 4.2
        import scipy.signal as sp

        fltrr = sp.firwin(
            127, color_low, fs=(315 / 88) * 4, window="boxcar", pass_zero="lowpass"
        )
        chrma_x = fltr_to_expr(chrma_x, fltrr)
        i, q = split_iq(chrma_x * pkpk)
        i, q = sm.eval_x([i, q])
    if color_notch:
        import scipy.signal as sp

        fltrr = sp.firwin(127, [4.5, 5], fs=(315 / 88) * 4, pass_zero="bandstop")
        chrma_x = fltr_to_expr(chrma_x, fltrr)
        i, q = split_iq(chrma_x * pkpk)
        i, q = sm.eval_x([i, q])

    if color_bp:
        i, q = tuple(fltr_to_expr(a, c_colorlp_b) for a in [i, q])
    i, q = sm.eval_x([i, q])

    u1, v1 = uv_from_iq(i, q, chroma_phase, chroma_gain)
    if keep_iq:
        u1, v1 = i, q
    if do_rgb:
        r, g, b = to_rgb(y1, u1, v1, consts)
        arra = [
            core.std.Expr(a, "x", format=vs.GRAY16)
            for a in [sm.eval_v(r), sm.eval_v(g), sm.eval_v(b)]
        ]
        xasd = join(arra, family=vs.ColorFamily.RGB)
    else:
        y, u, v = to_yuv(y1, u1, v1, consts)
        xasd = join(
            [
                a.std.Expr("x 256 * 256 *", format=vs.GRAY16)
                for a in [
                    sm.eval_v(y),
                    sm.eval_v(u),
                    sm.eval_v(v),
                ]
            ]
        ).std.SetFrameProps(_ColorRange=1)

    crop_params = {"left": 134, "right": 16, "top": 40, "bottom": 0}
    if crop:
        xasd = xasd.std.Crop(**crop_params)
    xasd = xasd.std.SetFrameProps(_SARDen=88, _SARNum=75)
    return xasd


def split_1d(_sm: StackableManager, vid_x: Stackable):
    return fltr_to_expr(vid_x, [-0.25, 0, 0.5, 0, -0.25])


def split_2d(
    sm: StackableManager,
    vid_x: Stackable,
    *,
    consts: CombConsts,
    chr1d: Optional[Stackable] = None,
) -> Stackable:
    if chr1d is None:
        chr1d = fltr_to_expr(vid_x, [-0.25, 0, 0.5, 0, -0.25])
        chr1d = sm.eval_x(chr1d)

    def temp1(x: Stackable, ofst: int):
        a = (x[0, 0].abs() - x[0, ofst].abs()).abs()
        b = (x[-1, 0].abs() - x[-1, ofst].abs()).abs()
        c = (x[0, 0].abs() + x[-1, ofst].abs()) * 0.1
        return a + b - c

    def temp2(pa: Stackable, consts=consts):
        return (Stackable.const(1) - (pa / consts.kRange)).clip(0.0, 1.0)

    def temp3(x: Stackable, pa: Stackable, op: Stackable):
        a = (op > (pa * 3)).iftrue(0, pa)

        b1 = (x[0, -2].abs() - x[0, 2].abs()).abs()
        b2 = ((x[0, 2] + x[0, -2]) * 0.2).abs()
        b = ((b1 - b2) <= 0).iftrue(1, pa)

        return ((op > 0) | (pa > 0)).iftrue(a, b)

    kp_o = temp1(chr1d, 2)
    kn_o = temp1(chr1d, -2)

    # kp_o,kn_o = sm.eval_x([kp_o,kn_o])
    kp_norm = temp2(kp_o)
    kn_norm = temp2(kn_o)
    kp_norm, kn_norm = sm.eval_x([kp_norm, kn_norm])

    kp = temp3(chr1d, kp_norm, kn_norm)
    kn = temp3(chr1d, kn_norm, kp_norm)
    kp, kn = sm.eval_x([kp, kn])

    def c_sc(kp_norm, kn_norm, kp, kn):
        a = (Stackable.const(2) / (kn + kp)).max(1)
        b = 1

        return ((kp_norm > 0) | (kn_norm > 0)).iftrue(a, b)

    def tc1(x: Stackable, kp, kn, sc):
        a = (x[0, 0] - x[0, -2]) * kn
        b = (x[0, 0] - x[0, 2]) * kp
        return ((a + b) * sc) / 4

    sc = c_sc(kp_norm, kn_norm, kp, kn)
    asd = tc1(chr1d, kp, kn, sc)
    return asd


def split_3d(
    sm, vid_x: Stackable, *, consts: CombConsts, chr2d_override: vs.VideoNode = None
):
    @dataclass
    class Reference:
        base: Stackable
        luma: Stackable
        chroma: Stackable

    def compare_candidate(ref: Reference, can: Reference):
        yPenalty = Stackable.const(0)
        for offset in range(-1, 2):
            yPenalty += (ref.luma - can.luma)[offset, 0].abs()
        yPenalty = (yPenalty / 3) / consts.irescale

        iqPenalty = Stackable.const(0)

        weights = [0.5, 1.0, 0.5]
        for offset in range(-1, 2):
            iqPenalty += (ref.chroma - (-can.chroma))[offset, 0].abs() * weights[
                offset + 1
            ]

        iqPenalty = ((iqPenalty / 2) / consts.irescale) * 0.28

        return yPenalty + iqPenalty

    def mkref(vid_x: Stackable, a: Stackable, offset: tuple[int, int] = (0, 0), sm=sm):
        #    return Reference(sm.eval_x(vid_x[offset]),sm.eval_x(vid_x[offset] - a[offset]),sm.eval_x(a[offset]))
        return Reference(vid_x[offset], vid_x[offset] - a[offset], a[offset])

    d1 = split_1d(sm, vid_x)
    d1, d1v = sm.eval(d1)

    d2 = split_2d(sm, vid_x, consts=consts, chr1d=d1)
    d2, d2v = sm.eval(d2)

    if chr2d_override:
        d2 = sm.add_clip(chr2d_override)

    vid_v = sm.eval_v(vid_x)

    LINE_BONUS = -2.0
    FIELD_BONUS = LINE_BONUS - 2.0
    FRAME_BONUS = FIELD_BONUS - 2.0

    origin_ref = mkref(vid_x, d2)

    d1_left = compare_candidate(origin_ref, mkref(vid_x, d2, (-2, 0)))
    d1_right = compare_candidate(origin_ref, mkref(vid_x, d2, (2, 0)))

    # up/down wrong
    d2_2up = compare_candidate(origin_ref, mkref(vid_x, d2, (0, 2))) + LINE_BONUS
    d2_2down = compare_candidate(origin_ref, mkref(vid_x, d2, (0, -2))) + LINE_BONUS

    # d2_1up   = compare_candidate(mkref(vid_x,d2),mkref(vid_x,d2,(0, 1))) + FIELD_BONUS
    # d2_1down = compare_candidate(mkref(vid_x,d2),mkref(vid_x,d2,(0,-1))) + FIELD_BONUS

    # d2_1   = ((phase_flip(vid_x) * phase_flip(sm.eval_x(vid_x[0,1]))) > 0).iftrue(d2_1down,d2_1up)
    # d2_1_v = ((phase_flip(vid_x) * phase_flip(sm.eval_x(vid_x[0,1]))) > 0).iftrue(
    #    (d2 - d2[0,-1]) / 2,
    #    (d2 - d2[0, 1]) / 2
    # )

    vx_nxt = sm.add_clip(vid_v[1:] + vid_v[-1])
    vx_prv = sm.add_clip(vid_v[0] + vid_v[:-1])

    d1_nxt = sm.add_clip(d1v[1:] + d1v[-1])
    d1_prv = sm.add_clip(d1v[0] + d1v[:-1])

    d2_nxt = sm.add_clip(d2v[1:] + d2v[-1])
    d2_prv = sm.add_clip(d2v[0] + d2v[:-1])

    d2_1up = compare_candidate(origin_ref, mkref(vid_x, d2, (0, 1))) + FIELD_BONUS
    d2_1down = compare_candidate(origin_ref, mkref(vid_x, d2, (0, -1))) + FIELD_BONUS

    d2_nxt_1up = (
        compare_candidate(origin_ref, mkref(vx_nxt, d2_nxt, (0, 1))) + FIELD_BONUS
    )
    d2_prv_1down = (
        compare_candidate(origin_ref, mkref(vx_prv, d2_prv, (0, -1))) + FIELD_BONUS
    )

    d2_1_cond = (phase_flip_props(vid_x) * phase_flip_props(sm.eval_x(vid_x[0, 1]))) < 0
    d2_1_1 = d2_1_cond.iftrue(d2_1up, d2_1down)
    d2_1_2 = d2_1_cond.iftrue(d2_prv_1down, d2_nxt_1up)

    d_1_v_1 = d2_1_cond.iftrue(
        (d1 - d1[0, 1]) / 2,
        (d1 - d1[0, -1]) / 2,
    )
    d_1_v_2 = d2_1_cond.iftrue(
        (d1 - d1_prv[0, -1]) / 2,
        (d1 - d1_nxt[0, 1]) / 2,
    )

    d3_nxt_p = compare_candidate(origin_ref, mkref(vx_nxt, d2_nxt)) + FRAME_BONUS
    d3_prv_p = compare_candidate(origin_ref, mkref(vx_prv, d2_prv)) + FRAME_BONUS

    smlst = Stackable.choose_smallest(
        sm,
        [
            (d1_left, d2),
            (d1_right, d2),
            (d2_2up, d2),
            (d2_2down, d2),
            (d2_1_1, d_1_v_1),
            (d2_1_2, d_1_v_2),
            (d3_nxt_p, (d1 - d1_nxt) / 2),
            (d3_prv_p, (d1 - d1_prv) / 2),
        ],
    )

    return smlst


def combyc_field_model(vid_v: vs.VideoNode, network_path=None, backend=None, **kwargs):
    from vsmlrt import inference, BackendV2

    if backend is None:
        backend2 = BackendV2.TRT()
    else:
        backend2 = backend

    if network_path is None:
        network_path = get_model_path("luma_sep_2dgray_fields.onnx")

    mdl_in = vid_v.std.SeparateFields(tff=True)

    lma_field = inference(
        mdl_in, network_path=network_path, backend=backend2
    ).std.DoubleWeave(tff=True)[::2]

    return comb_split_already(
        vid_v, core.akarin.Expr([vid_v, lma_field], "x y -"), **kwargs
    )


def combyc_frame_model(vid_v: vs.VideoNode, network_path=None, backend=None, **kwargs):
    from vsmlrt import inference, Backend

    mdl_in = vid_v
    if backend is None:
        backend = Backend.TRT
    if network_path is None:
        network_path = get_model_path("luma_sep_2d_frame_gray_gray_run2_latest.onnx")

    lma_field = inference(mdl_in, network_path=network_path, backend=backend)

    return comb_split_already(
        vid_v, core.akarin.Expr([vid_v, lma_field], "x y -"), **kwargs
    )


def comb_crop(vid_v: vs.VideoNode, crop=True):
    if not crop:
        return vid_v
    else:
        crop_params = {"left": 134, "right": 16, "top": 40, "bottom": 0}
        return vid_v.std.Crop(**crop_params)


def comb1d(vid_v: vs.VideoNode, **kwargs):
    sm = StackableManager()
    vid_x = sm.add_clip(vid_v)
    chrm2d = sm.eval_x(split_1d(sm, vid_x))

    return comb_split_already(vid_v, sm.eval_v(chrm2d), **kwargs)


def comb2d(vid_v: vs.VideoNode, consts=CombConsts(True), **kwargs):
    sm = StackableManager()
    vid_x = sm.add_clip(vid_v)
    chrm2d = sm.eval_x(split_2d(sm, vid_x, consts=consts))

    return comb_split_already(vid_v, sm.eval_v(chrm2d), consts=consts, **kwargs)


def comb2d_2(vid_v: vs.VideoNode, **kwargs):
    sm = StackableManager()
    vid_x = sm.add_clip(vid_v)

    a = (vid_x[0, 0] + vid_x[0, 1]) / 2
    b = (vid_x[0, 0] + vid_x[0, -1]) / 2
    lma = (oY % 2 == 0).iftrue(a, b)
    # lma =  (vid_x[0, 0] + vid_x[0, -2]) / 2
    # a =  (vid_x[0, 0] + vid_x[0, -2] ) / 2
    # b =  (vid_x[0, 0] + vid_x[0, -5]) / 2
    # lma = (oY  % 2 == 0).iftrue(a,b)

    # lma = sm.add_clip(vid_v.std.BoxBlur())

    chrma = sm.eval_x(vid_x - lma)
    chrm2d = sm.eval_x(chrma)

    return comb_split_already(vid_v, sm.eval_v(chrm2d), **kwargs)


def comb3d(
    vid_v: vs.VideoNode, *, consts=CombConsts(True), chr2d_override=None, **kwargs
):
    sm = StackableManager()
    vid_x = sm.add_clip(vid_v)
    chrm3d = sm.eval_x(
        split_3d(sm, vid_x, consts=consts, chr2d_override=chr2d_override)
    )

    return comb_split_already(vid_v, sm.eval_v(chrm3d), consts=consts, **kwargs)


def input_for_color_cnn(frames, x_offset=0):
    # y_offset=0 needs invert of the ifture aswell =
    fields_for_tbcc2 = ntsc_frames_to_fields(frames)
    fsc = COLOR_CARIER_FREQ_FLT
    sm = StackableManager()
    # we need to add because of x.
    gg = sm.add_clip(fields_for_tbcc2)
    cc = ((oX + x_offset) / (4 * fsc)) * 2 * math.pi * fsc + ((-90) * (math.pi / 180))
    swtch = (((oY) % 2) == 1).iftrue(-1.0, 1.0) * gg["fieldPhase"].switch(
        [
            (1, 1),
            (2, -1),
            (3, -1),
            (4, 1),
        ]
    )
    mod1 = sm.eval_v(cc.cos() * swtch)
    mod2 = sm.eval_v(cc.sin() * swtch)
    a = fields_for_tbcc2
    return mod1, mod2, a


def comb_color_cnn_color_notch(*args, **kwargs):
    ya = comb_color_cnn(*args, **kwargs)
    yb = comb_color_cnn(color_notch_lp(*args), **kwargs)
    return join(
        [
            split(ya)[0],
            split(yb)[1],
            split(yb)[2],
        ]
    )


def comb_color_cnn(
    frames,
    v2=False,
    perfield=True,
    network_path=None,
    consts=CombConsts(True),
    crop=False,
    bp_q=False,
    backend=None,
    chroma_phase=0.0,
    iq_cariers: Optional[tuple[vs.VideoNode, vs.VideoNode]] = None,
):
    from vsmlrt import inference, BackendV2

    if backend is None:
        backend = BackendV2.TRT()
    #        desired_crop_params = {
    #            "left": project.project["videoParameters"]["activeVideoStart"],
    #            "right": project.project["videoParameters"]["fieldWidth"] - project.project["videoParameters"]["activeVideoEnd"],
    #            #40 525
    #            "top": 39,
    #            "bottom": 1
    #        }

    # TODO: remove copy paste
    crop_params = {"left": 132, "right": 16, "top": 40, "bottom": 0}
    # hardcode ntscj for clippping only lowerbound
    # THIS clips color sometimes
    # frames = frames.std.Expr(f"x {consts.black16bIre_ntscj} max")
    frames = frames.akarin.Expr(f"X 454 > Y 524 >= and {consts.black16bIre_ntscj} x ?")
    if crop:
        frames = (
            frames.resize.Bicubic(format=vs.GRAY16, range_in=1, range=1)
            .fb.FillBorders(
                mode="mirror",
                left=crop_params["left"],
                right=crop_params["right"],
                top=crop_params["top"],
            )
            .resize.Bicubic(format=vs.GRAYS, range_in=1, range=1)
        )
    #    frames = frames.std.Crop(**crop_params)
    mod1, mod2, a = input_for_color_cnn(frames, x_offset=0)
    if iq_cariers is not None:
        mod1, mod2 = iq_cariers

    mdlin = join([a, mod1, mod2])
    if not perfield:
        mdlin = mdlin.std.DoubleWeave(tff=True)[::2]
    # TODO lookputable
    if network_path is None:
        # (v2,perfield)
        network_path = get_model_path(
            {
                (True, True): "color_cnn_v2_alot.onnx",
                (True, False): "color_cnn_frames_434k.onnx",
                (False, True): "color_cnn_1031640.onnx",
                # untrainewd
            }[v2, perfield]
        )
    rawout = inference(mdlin, network_path=network_path, backend=backend)

    #    if crop:
    #        rawout = rawout.std.Crop(left=desired_crop_params["left"] % 4,top=desired_crop_params["top"] % 4)

    rawout = split(rawout)
    sm = StackableManager()
    y, i, q = tuple([sm.add_clip(rawout[a]) for a in range(3)])

    u1, v1 = uv_from_iq(i, q, chroma_gain=1, chromaPhase=chroma_phase)

    y, u, v = to_yuv(y, u1, v1, consts=consts)

    outcl = join(
        [
            a.std.Expr("x 256 * 256 *", format=vs.GRAY16)
            for a in [
                sm.eval_v(y),
                sm.eval_v(u),
                sm.eval_v(v),
            ]
        ]
    ).std.SetFrameProps(_ColorRange=1)
    if perfield:
        outcl = outcl.std.DoubleWeave(tff=True)[::2]

    xasd = outcl
    if crop:
        xasd = xasd.std.Crop(**crop_params).std.Crop(left=2).std.AddBorders(top=2)
    xasd = xasd.std.SetFrameProps(_SARDen=88, _SARNum=75)
    from vstools import Matrix, Transfer, Primaries

    xasd = Matrix.SMPTE170M.apply(Transfer.BT601.apply(Primaries.BT601_525.apply(xasd)))

    return xasd


### i dont think i implemtned it properly
###add #include <unistd.h>
###gcc tensorflow_c_predict.c -I/usr/include/tensorflow -Wno-implicit-function-declaration -ltensorflow -o cvbs_ai_decode
##def cvbs_ai(fields: vs.VideoNode,**kwargs):
##    import subprocess
##    assert (fields.width,fields.height) == (910,263)
##    assert fields.format.id==vs.GRAY16
##    croped_fields = fields.std.Crop(top=21)
##
##    MDIR = get_model_path("cvbs-ai-decode")
##
##    ps = subprocess.Popen(('/lib64/ld-linux-x86-64.so.2', '--library-path', "./", f"{MDIR}/cvbs_ai_decode", "-m" f"{MDIR}/model/ntsc_yc", "-o", "yc"), stdout=subprocess.PIPE, stdin=subprocess.PIPE,cwd=get_model_path(""))
##    def ModFrame(n,f,stdin=ps.stdin,stdout=ps.stdout):
##        import numpy as np
##        f2 = f[1].copy()
##        f = f[0]
##        assert (f.width,f.height) == (910,242)
##
##        lkll = bytes(f[0])
##        stdin.write(lkll)
##        stdin.flush()
##
##        wptr = f2.get_write_ptr(0)
##        import ctypes as ct
##        strd = f2.get_stride(0)
##
##        lale = 242 * strd * 2 * 2
##        wptr = ct.cast(wptr, ct.POINTER(ct.c_char * (lale) ))#* f.height * f.get_stride(0) * 2))
##
##        raro = stdout.read(242 * 792 * 2 * 2)
##        fbb = np.frombuffer(raro,dtype=np.uint16)
##        lal1 = fbb[::2]
##        rar = lal1.tobytes()
##        lal2 = fbb[1::2]
##        rar2 = lal2.tobytes()
##
##        for h in range(f2.height // 2):
##            stst = h * strd
##            osd = h * 792 * 2
##            ll = 792*2
##            wptr.contents[stst:stst + ll] = rar[osd:osd+ll]
##
##        for h in range(f2.height // 2 ,f2.height):
##            stst = h * strd
##            osd = (h - f2.height // 2) * 792 * 2
##            ll = 792*2
##            wptr.contents[stst:stst + ll] = rar2[osd:osd+ll]
##
##        return f2
##
##    outcl = core.std.BlankClip(fields,width=792,height=242 * 2)
##    tbcc2 = core.std.ModifyFrame(outcl,clips=[croped_fields,outcl],selector=ModFrame)
##    chrma = tbcc2.std.Crop(top=242)
##    lma = tbcc2.std.Crop(bottom=242)
##
##
##    lla =  lma.std.AddBorders(left=118,top=21)
##    llc = chrma.std.AddBorders(left=118,top=21)
##
##    vid_v1 = lla.std.DoubleWeave(tff=True)[::2]
##    vid_v2 = llc.std.DoubleWeave(tff=True)[::2]
##
##    return comb_split_already(
##        ntsc_fields_to_frames(fields).resize.Bicubic(format=vs.GRAYS,range_in=1,range=1),
##        chrma=vid_v2.resize.Bicubic(format=vs.GRAYS,range_in=1,range=1),
##        lma_override=vid_v1.resize.Bicubic(format=vs.GRAYS,range_in=1,range=1),
##        **kwargs
##    )
