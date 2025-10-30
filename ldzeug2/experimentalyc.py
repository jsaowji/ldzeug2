import torch
from torch import nn


class FullModelExperimental(torch.nn.Module):
    def __init__(self, num_feat=64, num_conv=16, upscale=2):
        super(FullModelExperimental, self).__init__()
        num_in_ch = 1
        num_out_ch = 3
        kernel_size = 3
        strid = 1
        padd = "same"
        bias = True
        self.upscale = upscale

        self.num_feat = num_feat
        self.num_conv = num_conv

        self.body1 = nn.ModuleList()
        self.body2 = nn.ModuleList()
        self.body3 = nn.ModuleList()

        self.body1.append(
            nn.Conv2d(num_in_ch, num_feat, kernel_size, strid, padd, bias=bias)
        )
        self.body1.append(nn.PReLU(num_parameters=num_feat))

        self.params1 = torch.nn.parameter.Parameter(torch.randn(num_feat))
        self.params2 = torch.nn.parameter.Parameter(torch.randn(num_feat))

        for _ in range(num_conv):
            self.body2.append(
                nn.Conv2d(num_feat, num_feat, kernel_size, strid, padd, bias=bias)
            )
            self.body2.append(nn.PReLU(num_parameters=num_feat))
        self.body3.append(
            nn.Conv2d(
                num_feat,
                num_out_ch * upscale * upscale,
                kernel_size,
                strid,
                padd,
                bias=bias,
            )
        )
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        batch_cnt, planes, w, h = x.shape

        inbuf = x[:, 0, :, :].unsqueeze(1)
        i_mlt = x[:, 1, :, :].unsqueeze(1)
        q_mlt = x[:, 2, :, :].unsqueeze(1)

        out = inbuf
        for i in range(0, len(self.body1)):
            out = self.body1[i](out)

        for i in range(self.num_conv):
            out = self.body2[i](out)
            if i == self.num_conv // 2:
                p = self.params1.reshape((1, self.num_feat, 1, 1))
                out = ((1.0 - p) * out) + ((p) * (out * i_mlt))

                p = self.params2.reshape((1, self.num_feat, 1, 1))
                out = ((1.0 - p) * out) + ((p) * (out * q_mlt))

        for i in range(0, len(self.body3)):
            out = self.body3[i](out)
        out = self.upsampler(out)
        return out
