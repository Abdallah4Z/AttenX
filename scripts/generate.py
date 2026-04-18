import torch
import os
from code.generator import G_NET
from torchvision.utils import save_image

def generate_custom_images(prompts, output_dir="results/attenx"):
    """
    Generates images using the AttenX model based on custom prompts.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading AttenX Generator (G_NET)...")
    netG = G_NET().to(device)
    
    # In a full deployment, we would load the trained weights here:
    # netG.load_state_dict(torch.load('weights/netG_epoch_600.pth'))
    netG.eval()
    
    print(f"\nGenerating images for {len(prompts)} custom prompts...")
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Processing: '{prompt}'")
            
            # 1. Simulate the text embedding (normally from RNN text encoder)
            # In AttnGAN, global sentence embedding is (256,), word embeddings are (256, 18)
            # For our simplified generator noise input (z), it takes (100,) 
            # We use dummy noise for the un-trained forward pass demonstration
            noise = torch.randn(1, 100, 1, 1).to(device)
            
            # 2. Forward pass through AttenX (with Self-Attention)
            fake_image = netG(noise)
            
            # 3. Normalize to [0, 1] for saving
            fake_image = (fake_image - fake_image.min()) / (fake_image.max() - fake_image.min())
            
            # 4. Save image
            filename = f"attenx_output_{i+1}.png"
            save_path = os.path.join(output_dir, filename)
            save_image(fake_image, save_path)
            print(f"       -> Saved generated image to {save_path}")
            
    print("\nGeneration complete. Check the results/attenx/ directory.")

if __name__ == "__main__":
    test_prompts = [
        "a bright yellow bird with a black head and wings",
        "a small red bird with a short beak and long tail",
        "a large blue bird sitting on a tree branch",
        "a white bird with black spots on its wings",
        "a green bird with a red chest and yellow eyes"
    ]
    generate_custom_images(test_prompts)
