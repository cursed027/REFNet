import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): 
        return self.net(x)

class GatedRefinerUNet(nn.Module):
    def __init__(self):
        super().__init__()
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
        
        # Initialized at 0.5 to prevent aggressive early-training color shifts
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        I_h = x[:, 3:6, :, :] 
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d3 = self.conv_up1(torch.cat([x3, self.up1(x4)], dim=1))
        d2 = self.conv_up2(torch.cat([x2, self.up2(d3)], dim=1))
        d1 = self.conv_up3(torch.cat([x1, self.up3(d2)], dim=1))
        
        out = self.outc(d1)
        
        R = out[:, 0:3, :, :]                 
        A = torch.sigmoid(out[:, 3:4, :, :])  
        
        # --- THE SAFETY CLAMP: ReLU prevents negative sign flips ---
        I_final = I_h + torch.relu(self.alpha) * (A * R)
        
        # Return R so we can track its magnitude in the training loop
        return I_final, A, R
