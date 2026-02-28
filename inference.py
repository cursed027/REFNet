import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import RefinerDataset
from network import GatedRefinerUNet
import torchvision.transforms.functional as TF
from math import log10
from tqdm import tqdm
import cv2
import numpy as np

def load_config(yml_path):
    with open(yml_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to val config YAML')
    args = parser.parse_args()

    config = load_config(args.opt)
    device = torch.device(config['device'])
    
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Initializing Validation with Checkpoint: {config['ckpt_path']}")
    
    # Load Dataset
    val_set = RefinerDataset(
        config['data']['degraded'], 
        config['data']['histo'], 
        config['data']['gt'], 
        crop_size=None
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    
    # Load Model
    model = GatedRefinerUNet().to(device)
    checkpoint = torch.load(config['ckpt_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    total_psnr = 0.0
    
    print("🔍 Running Validation...")
    with torch.no_grad():
        for idx, (inputs, gt) in enumerate(tqdm(val_loader, desc="Evaluating")):
            inputs, gt = inputs.to(device), gt.to(device)
            
            # Forward pass
            I_final, A, R = model(inputs)
            
            # Calculate PSNR
            mse = torch.mean((I_final - gt)**2)
            if mse.item() > 0:
                total_psnr += 10 * log10(1 / mse.item())
            
            # Save Image if requested
            if config['save_images']:
                img_name = val_set.image_names[idx]
                
                # Convert to numpy [H, W, C] and BGR for OpenCV
                out_np = I_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(save_dir, img_name), out_bgr)
                
                # Optional: Save the Attention Map to visualize where it's looking
                if config.get('save_attention', False):
                    att_np = A.squeeze(0).squeeze(0).cpu().numpy()
                    att_np = (att_np * 255.0).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, f"att_{img_name}"), att_np)

    avg_psnr = total_psnr / len(val_loader) if len(val_loader) > 0 else 0
    print(f"\n✅ Validation Complete!")
    print(f"📊 Average PSNR: {avg_psnr:.3f} dB")
    print(f"💾 Images saved to: {save_dir}")

if __name__ == "__main__":
    main()
