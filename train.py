import os
import yaml
import argparse
import datetime
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RefinerDataset
from network import GatedRefinerUNet
from loss import RefinerLoss
from math import log10

def load_config(yml_path):
    with open(yml_path, 'r') as f:
        return yaml.safe_load(f)

def setup_experiment(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{config['name']}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    
    log_file = os.path.join(exp_dir, "train.log")
    
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
            
    log(f"🚀 Starting Experiment: {config['name']}")
    log(f"📁 Log Directory: {exp_dir}")
    return exp_dir, log

def format_time(seconds):
    """Converts seconds to an aesthetic hh:mm:ss format"""
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    config = load_config(args.opt)
    exp_dir, log = setup_experiment(config)
    device = torch.device(config['device'])

    # --- DATALOADERS ---
    train_set = RefinerDataset(
        config['data']['train']['degraded'], 
        config['data']['train']['histo'], 
        config['data']['train']['gt'], 
        crop_size=config['data']['train']['crop_size']
    )
    
    train_loader = DataLoader(
        train_set, batch_size=config['data']['train']['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True, 
        prefetch_factor=4, persistent_workers=True
    )
    
    val_set = RefinerDataset(
        config['data']['val']['degraded'], 
        config['data']['val']['histo'], 
        config['data']['val']['gt'], 
        crop_size=None
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # --- MODEL & OPTIMIZER ---
    model = GatedRefinerUNet().to(device)
    criterion = RefinerLoss(device, config['loss']['lambda_lpips'], config['loss']['lambda_sparse'])
    optimizer = optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['total_iters'])

    start_iter = 0

    # --- RESUME LOGIC (Now checking the YAML config) ---
    resume_path = config.get('resume_state', None)
    if resume_path and os.path.exists(resume_path):
        log(f"🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_iter = checkpoint['iteration']

    # --- TRAINING LOOP ---
    log("⚙️  Initiating Training Loop...")
    model.train()
    train_iter = iter(train_loader)
    
    start_time = time.time()
    iter_start_time = time.time()
    
    for current_iter in range(start_iter + 1, config['total_iters'] + 1):
        try:
            inputs, gt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, gt = next(train_iter)
            
        inputs, gt = inputs.to(device), gt.to(device)
        
        optimizer.zero_grad()
        I_final, A = model(inputs)
        
        loss, l_char, l_lpips, l_sparse = criterion(I_final, gt, A)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- LOGGING & ETA CALCULATOR ---
        if current_iter % config['print_freq'] == 0:
            iter_time = (time.time() - iter_start_time) / config['print_freq']
            iters_left = config['total_iters'] - current_iter
            eta_seconds = iters_left * iter_time
            
            log(f"Iter [{current_iter:05d}/{config['total_iters']}] "
                f"| Loss: {loss.item():.4f} | Charb: {l_char.item():.4f} | LPIPS: {l_lpips.item():.4f} | Sparse(A): {l_sparse.item():.4f} "
                f"| {iter_time:.2f} s/iter | ETA: {format_time(eta_seconds)}")
            
            iter_start_time = time.time() # Reset timer for next block

        # --- VALIDATION ---
        if current_iter % config['val_freq'] == 0:
            model.eval()
            val_loss, val_psnr = 0.0, 0.0
            with torch.no_grad():
                for v_in, v_gt in val_loader:
                    v_in, v_gt = v_in.to(device), v_gt.to(device)
                    v_final, _ = model(v_in)
                    mse = torch.mean((v_final - v_gt)**2)
                    # Prevent math domain error on perfect identical noise
                    if mse.item() > 0:
                        val_psnr += 10 * log10(1 / mse.item())
                    
            avg_psnr = val_psnr / len(val_loader) if len(val_loader) > 0 else 0
            log(f"🔍 [VALIDATION] Iter: {current_iter} | PSNR: {avg_psnr:.2f} dB")
            model.train()

        # --- CHECKPOINTING ---
        if current_iter % config['save_freq'] == 0:
            ckpt_path = os.path.join(exp_dir, "checkpoints", f"refiner_iter_{current_iter}.pth")
            torch.save({
                'iteration': current_iter,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }, ckpt_path)
            log(f"💾 Saved Checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    log(f"✅ Training Complete. Total Time: {format_time(total_time)}")

if __name__ == "__main__":
    main()
