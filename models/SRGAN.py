# 3D pytorch implementation of SRGAN, only generator part
import torch.nn as nn
import math
import torch.nn.functional as F

class up_blk(nn.Module):
    def __init__(self, in_ch, out_ch, act, up,):
        super(up_blk, self).__init__()
        self.up = up
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, ),
            act,
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, ),
            act,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=self.up, mode='trilinear', align_corners=True, )
        x = self.conv2(x)
        return x

class generator(nn.Module):
    def __init__(self, conf=None):
        super(generator, self).__init__()
        self.scale = conf.scale
        self.inplanes = conf.inplanes
        self.res_blk_num = conf.res_blk_num
        self.up_blk_num = int(math.log2(self.scale))
        self.activation = conf.activation
        self.normalization = conf.normalization
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        if self.normalization == 'Batch':
            self.norm = nn.BatchNorm3d
        elif self.normalization == 'Instance':
            self.norm = nn.InstanceNorm3d
        self.channel_reduce_factor = conf.channel_reduce_factor

        # input block
        self.in_blk = nn.Sequential(
            nn.Conv3d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, ),
            self.norm(self.inplanes),
            self.act,
        )
        # # residual block
        self.res_blk_list = nn.ModuleList(self.make_res_blk() for _ in range(self.res_blk_num))
        self.res_blk_last = nn.Sequential(
            nn.Conv3d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, ),
            self.norm(self.inplanes),
        )
        # upsampling block
        self.up_blk_list = nn.ModuleList()
        for i in range(self.up_blk_num):
            if i<2:
                channel_in = self.inplanes//(self.channel_reduce_factor**i)
                channel_out = self.inplanes//(self.channel_reduce_factor**(i+1))
            else:
                channel_in =  self.inplanes//(self.channel_reduce_factor**2)
                channel_out = self.inplanes // (self.channel_reduce_factor**2)
            self.up_blk_list.append(up_blk(channel_in,channel_out,self.act,2))
        # output block
        self.out_blk = nn.Sequential(
            nn.Conv3d(channel_out, 1, kernel_size=3, stride=1, padding=1, ),
        )

    def make_res_blk(self,):
        res_blk = nn.Sequential(
            nn.Conv3d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, ),
            self.norm(self.inplanes),
            self.act,
            nn.Conv3d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, ),
            self.norm(self.inplanes),
        )
        return res_blk

    def forward(self,x):
        # input stage
        x = self.in_blk(x)

        # residual stage
        final_residual = x
        residual = x
        for i, blk in enumerate(self.res_blk_list):
            x = self.res_blk_list[i](x)
            x = x + residual
            x = self.act(x)
            residual = x
        x = self.res_blk_last(x)
        x = x + final_residual
        x = self.act(x)

        # upsampling stage
        for i, blk in enumerate(self.up_blk_list):
            x = self.up_blk_list[i](x)

        # output stage
        x = self.out_blk(x)
        return x
