import torch
import torch.nn as nn
import lpips

class RefinerLoss(nn.Module):
    def __init__(self, device, lambda_lpips=0.1, lambda_sparse=0.005):
        super().__init__()
        # NTIRE standard is AlexNet
        self.lpips_net = lpips.LPIPS(net='alex').to(device)
        self.lpips_net.eval()
        for param in self.lpips_net.parameters():
            param.requires_grad = False
            
        self.lambda_lpips = lambda_lpips
        self.lambda_sparse = lambda_sparse
        self.eps = 1e-3

    def forward(self, I_final, I_gt, A):
        # 1. Charbonnier (Fidelity)
        loss_char = torch.mean(torch.sqrt((I_final - I_gt)**2 + self.eps**2))
        
        # 2. LPIPS (Perceptual)
        I_final_norm = (I_final * 2.0) - 1.0
        I_gt_norm = (I_gt * 2.0) - 1.0
        loss_lpips = self.lpips_net(I_final_norm, I_gt_norm).mean()
        
        # 3. Sparsity (Gate Control)
        loss_sparse = torch.mean(torch.abs(A))
        
        total = loss_char + (self.lambda_lpips * loss_lpips) + (self.lambda_sparse * loss_sparse)
        return total, loss_char, loss_lpips, loss_sparse

