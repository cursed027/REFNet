import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.net(x)

class GatedRefinerUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 9 in -> 32 -> 64 -> 128 -> 256
        self.inc = DoubleConv(9, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        I_h = x[:, 3:6, :, :] # Extract Clean Histoformer Output
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d3 = self.conv_up1(torch.cat([x3, self.up1(x4)], dim=1))
        d2 = self.conv_up2(torch.cat([x2, self.up2(d3)], dim=1))
        d1 = self.conv_up3(torch.cat([x1, self.up3(d2)], dim=1))
        
        out = self.outc(d1)
        
        # --- GATING ---
        R = out[:, 0:3, :, :]                 # Texture Residual
        A = torch.sigmoid(out[:, 3:4, :, :])  # Attention Gate [0,1]
        
        I_final = I_h + (A * R)
        return I_final, A
