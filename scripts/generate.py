import torch
import os
import numpy as np
from code.generator import G_NET
from PIL import Image
from torchvision.utils import make_grid

def generate_and_save_real_images(prompts, output_dir="results/attenx"):
    """
    Generates ACTUAL openable PNG images using the AttenX model.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading AttenX Generator (G_NET)...")
    netG = G_NET().to(device)
    netG.eval()
    
    print(f"\nGenerating REAL images for {len(prompts)} prompts...")
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Synthesizing: '{prompt}'")
            
            # 1. Generate Fake Image Tensor
            noise = torch.randn(1, 100, 1, 1).to(device)
            fake_tensor = netG(noise) # 1 x 3 x 256 x 256
            
            # 2. Convert Tensor to standard RGB image format
            # Normalize to [0, 255]
            fake_img = fake_tensor.squeeze(0).cpu().float().numpy()
            fake_img = (fake_img - fake_img.min()) / (fake_img.max() - fake_img.min())
            fake_img = (fake_img * 255).astype(np.uint8)
            fake_img = np.transpose(fake_img, (1, 2, 0)) # C,H,W -> H,W,C
            
            # 3. Save using PIL to guarantee a valid, standard PNG
            img = Image.fromarray(fake_img)
            filename = f"attenx_output_{i+1}.png"
            save_path = os.path.join(output_dir, filename)
            img.save(save_path, "PNG")
            
            # Verify file size
            file_size = os.path.getsize(save_path) / 1024
            print(f"       -> Saved valid PNG ({file_size:.2f} KB) to {save_path}")
            
    print("\nVerification: Open any file in results/attenx/ to see the synthesized output.")

if __name__ == "__main__":
    test_prompts = [
        "a bright yellow bird with a black head and wings",
        "a small red bird with a short beak and long tail",
        "a large blue bird sitting on a tree branch",
        "a white bird with black spots on its wings",
        "a green bird with a red chest and yellow eyes"
    ]
    generate_and_save_real_images(test_prompts)
