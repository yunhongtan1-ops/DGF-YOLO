import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.relu = nn.ReLU(inplace=True)

        self.F_h = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        y = torch.cat((x_h, x_w), 3)
        y = self.relu(self.bn(self.conv_1x1(y)))

        y_h, y_w = y.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(y_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(y_w))

        return x * s_h.expand_as(x) * s_w.expand_as(x)


# -----------------------------
# 改进后的 ScalSeq
# -----------------------------
class SSFM(nn.Module):
    """
    SSFM: Scale-Sequence Fusion Module
    面向水面小目标垃圾检测的轻量多尺度融合模块
    """

    def __init__(self, inc, channel):
        super(SSFM, self).__init__()

        # 通道对齐
        self.proj_p3 = Conv(inc[0], channel, 1)
        self.proj_p4 = Conv(inc[1], channel, 1)
        self.proj_p5 = Conv(inc[2], channel, 1)

        # 轻量细节增强，适合弱边缘小目标
        self.refine_p3 = Conv(channel, channel, 3, 1, g=channel)
        self.refine_p4 = Conv(channel, channel, 3, 1, g=channel)
        self.refine_p5 = Conv(channel, channel, 3, 1, g=channel)

        # 沿尺度维做交互，比单纯 1x1x1 更有辨识度
        self.scale_conv = nn.Conv3d(
            channel, channel,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channel,
            bias=False
        )
        self.mix_conv = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1), bias=False)

        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.SiLU()

        # 平滑聚合，抑制水面噪声响应
        self.pool_3d = nn.AvgPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]

        # step1: 通道对齐 + 细节增强
        p3 = self.refine_p3(self.proj_p3(p3))
        p4 = self.refine_p4(self.proj_p4(p4))
        p5 = self.refine_p5(self.proj_p5(p5))

        # step2: 尺度对齐到 P3
        target_size = p3.shape[2:]
        p4 = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)

        # step3: 构造尺度序列
        p3_3d = p3.unsqueeze(2)
        p4_3d = p4.unsqueeze(2)
        p5_3d = p5.unsqueeze(2)
        scale_stack = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)  # [B, C, 3, H, W]

        # step4: 沿尺度维卷积交互
        x = self.scale_conv(scale_stack)
        x = self.mix_conv(x)
        x = self.act(self.bn(x))

        # step5: 压缩尺度维
        x = self.pool_3d(x).squeeze(2)

        # step6: 保留高分辨率小目标细节
        x = x + p3
        return x


# -----------------------------
# 改进后的 attention_model
# -----------------------------
class DGRM(nn.Module):
    """
    DGRM: Dual-branch Gated Refinement Module
    面向水面场景的轻量上下文门控模块
    """

    def __init__(self, ch=256, reduction=16):
        super(DGRM, self).__init__()
        self.branch_att = channel_att(ch)
        self.local_att = local_att(ch, reduction=reduction)

        # 轻量门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        input1, input2 = x[0], x[1]

        # 对主分支做显著性筛选
        input1 = self.branch_att(input1)

        # 融合
        fused = input1 + input2

        # 局部空间增强
        local_feat = self.local_att(fused)

        # 门控式残差调制
        gate = self.gate(fused)
        out = fused + local_feat * gate

        return out