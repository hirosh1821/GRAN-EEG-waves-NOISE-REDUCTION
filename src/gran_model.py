import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mx = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_gate = self.fc(self.avg(x).squeeze(-1))
        max_gate = self.fc(self.mx(x).squeeze(-1))
        weights = self.sig(avg_gate + max_gate)
        return x * weights.unsqueeze(-1)


class DilatedMultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        branch_channels = channels // 4
        self.d1 = nn.Conv1d(channels, branch_channels, 3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv1d(channels, branch_channels, 3, padding=2, dilation=2, bias=False)
        self.d4 = nn.Conv1d(channels, branch_channels, 3, padding=4, dilation=4, bias=False)
        self.d8 = nn.Conv1d(channels, branch_channels, 3, padding=8, dilation=8, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * 4, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        multiscale = torch.cat([self.d1(x), self.d2(x), self.d4(x), self.d8(x)], dim=1)
        return self.fuse(multiscale) + x


class GRANModel(nn.Module):
    def __init__(self, n_channels=3, base_ch=32):
        super().__init__()
        c = base_ch
        self.input_proj = nn.Sequential(nn.Conv1d(n_channels, c, 7, padding=3, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.enc1 = ResBlock1D(c, c, stride=2)
        self.enc2 = ResBlock1D(c, c * 2, stride=2)
        self.enc3 = ResBlock1D(c * 2, c * 4, stride=2)
        self.enc4 = ResBlock1D(c * 4, c * 8, stride=2)
        self.bottleneck = nn.Sequential(DilatedMultiScaleBlock(c * 8), SqueezeExcitation(c * 8))
        self.up4 = nn.ConvTranspose1d(c * 8, c * 4, 4, stride=2, padding=1)
        self.dec4 = ResBlock1D(c * 8, c * 4)
        self.up3 = nn.ConvTranspose1d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec3 = ResBlock1D(c * 4, c * 2)
        self.up2 = nn.ConvTranspose1d(c * 2, c, 4, stride=2, padding=1)
        self.dec2 = ResBlock1D(c * 2, c)
        self.up1 = nn.ConvTranspose1d(c, c, 4, stride=2, padding=1)
        self.dec1 = ResBlock1D(c * 2, c)
        self.output_head = nn.Conv1d(c, n_channels, 1)

    def forward(self, x):
        s0 = self.input_proj(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        e4 = self.enc4(s3)
        b = self.bottleneck(e4)
        d4 = self.dec4(torch.cat([self.up4(b), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1))
        return x + self.output_head(d1)
