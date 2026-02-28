import os
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

def create_fake_dataset(base_dir, num_images=20):
    folders = ['degraded', 'histo_out', 'gt']
    for f in folders:
        os.makedirs(os.path.join(base_dir, f), exist_ok=True)
        
    print(f"Generating {num_images} fake tensors...")
    for i in tqdm(range(num_images)):
        # Using the naming convention from your screenshot
        filename = f"day_00001_{i:05d}_blur.png" 
        
        # Create random 720x480 noise tensors
        for f in folders:
            noise = torch.rand(3, 480, 720)
            img = TF.to_pil_image(noise)
            img.save(os.path.join(base_dir, f, filename))
            
    print("Fake dataset ready!")

if __name__ == "__main__":
    create_fake_dataset("./fake_data/train")
    create_fake_dataset("./fake_data/val", num_images=5)
    
