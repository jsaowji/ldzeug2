from fractions import Fraction

__all__ = [
    "CombConsts",
    "COLOR_CARIER_FREQ_FLT",
    "COLOR_CARIER_FREQ_FRQ",
]

# constants for range 0-1
COLOR_CARIER_FREQ_FLT = (315 / 88) * 1_000_000
COLOR_CARIER_FREQ_FRQ = Fraction(315, 88) * 1_000_000


class CombConsts:
    def __init__(self, ntscj=True):
        self.Y_MIN = 1.0 / 256.0
        self.Y_ZERO = 16.0 / 256.0
        self.Y_SCALE = 219.0 / 256.0
        self.Y_MAX = 254.75 / 256.0
        self.C_MIN = 1.0 / 256.0
        self.C_ZERO = 128.0 / 256.0
        self.C_SCALE = 112.0 / 256.0
        self.C_MAX = 254.75 / 256.0

        self.ONE_MINUS_Kb = 1.0 - 0.114
        self.ONE_MINUS_Kr = 1.0 - 0.299
        self.kB = 0.49211104112248356308804691718185
        self.kR = 0.87728321993817866838972487283129

        self.black16bIre_ntscj = 15360.0 / (256 * 256)
        self.black16bIre_ntsc = 0x4680 / (256 * 256)

        # ntsc-j i believe
        # 0x3c66 once or 0x3c00
        self.black16bIre = self.black16bIre_ntscj if ntscj else self.black16bIre_ntsc

        self.white16bIre = 51200.0 / (256 * 256)

        self.yOffset = self.black16bIre
        self.yRange = self.white16bIre - self.black16bIre

        self.irescale = (self.white16bIre - self.black16bIre) / 100
        self.kRange = 45 * self.irescale

        self.uvRange = self.yRange

        self.yScale = self.Y_SCALE / self.yRange
        self.cbScale = (self.C_SCALE / (self.ONE_MINUS_Kb * self.kB)) / self.uvRange
        self.crScale = (self.C_SCALE / (self.ONE_MINUS_Kr * self.kR)) / self.uvRange
