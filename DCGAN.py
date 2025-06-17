import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Dataset Class
class SketchFaceDataset(Dataset):
    def __init__(self, sketch_dir, face_dir, size=256):
        # Fetch all image filenames
        sketch_files = sorted(os.listdir(sketch_dir))
        face_files = sorted(os.listdir(face_dir))

        # Match files using numeric portions of filenames
        sketch_map = {s.split('-')[1]: s for s in sketch_files}
        face_map = {f.split('-')[1]: f for f in face_files}
        matching_keys = set(sketch_map.keys()) & set(face_map.keys())

        # Get full file paths
        self.sketches = [os.path.join(sketch_dir, sketch_map[k]) for k in matching_keys]
        self.faces = [os.path.join(face_dir, face_map[k]) for k in matching_keys]
        self.size = size
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, index):
        sk = cv2.imread(self.sketches[index])
        fc = cv2.imread(self.faces[index])

        if sk is None or fc is None:
            print(f"Skipping bad image: {self.sketches[index]} or {self.faces[index]}")
            return self.__getitem__((index + 1) % len(self.sketches))

        sk = cv2.cvtColor(sk, cv2.COLOR_BGR2RGB)
        fc = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)
        sk = cv2.resize(sk, (self.size, self.size))
        fc = cv2.resize(fc, (self.size, self.size))

        return self.tf(sk), self.tf(fc)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

# Post-processing Functions
def post_process(image_tensor):
    """Apply post-processing to enhance the generated face."""
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Step 1: Exposure adjustment
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    exposure_factor = 0.85
    v = np.clip(v * exposure_factor, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])
    current_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)

    # Step 2: Sharpening
    float_image = current_image.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(float_image, (0, 0), sigmaX=2.0, sigmaY=2.0)
    detail = float_image - blurred
    clarity_alpha = 1.2
    sharpened = float_image + detail * clarity_alpha
    sharpened = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
    current_image = sharpened

    # Step 3: Edge-preserving smoothing
    smoothed = cv2.bilateralFilter(current_image, d=9, sigmaColor=30, sigmaSpace=30)
    current_image = smoothed

    # Step 4: Fix overexposed areas
    lab_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2LAB)
    lower_blue_lab = np.array([150, 0, 0])
    upper_blue_lab = np.array([255, 120, 120])
    mask = cv2.inRange(lab_image, lower_blue_lab, upper_blue_lab)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted_image = cv2.inpaint(current_image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Step 5: Contrast enhancement
    lab = cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge([cl, a, b])
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return final_image

def preprocess_sketch(image_path, device):
    """Preprocess a sketch image for the model."""
    sketch = cv2.imread(image_path)
    if sketch is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
    sketch = cv2.resize(sketch, (256, 256))
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    return tf(sketch).unsqueeze(0).to(device)

def generate_face(sk_tensor, generator):
    """Generate a face from a sketch tensor."""
    with torch.no_grad():
        generated_face = generator(sk_tensor).cpu().squeeze()
    generated_face = (generated_face + 1) / 2
    return generated_face

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy().transpose(1, 2, 0)
        img2 = img2.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Ensure the images are in range [0, 1]
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0
    
    return ssim(img1, img2, channel_axis=2, data_range=1.0)

def save_test_samples(generator, test_loader, device, epoch, output_dir):
    """Save test samples during training."""
    generator.eval()
    with torch.no_grad():
        for i, (sketch, real) in enumerate(test_loader):
            if i >= 5:  # Save first 5 test samples
                break
                
            sketch = sketch.to(device)
            fake = generator(sketch)
            
            # Calculate SSIM
            ssim_val = calculate_ssim(fake[0], real[0])
            
            # Create a grid of sketch, generated, and real images
            comparison = torch.cat([sketch[0], fake[0], real[0]], dim=2)
            save_image(comparison, 
                      os.path.join(output_dir, f'test_epoch_{epoch}_sample_{i}_ssim_{ssim_val:.3f}.png'),
                      normalize=True)
    generator.train()

def plot_losses(g_losses, d_losses, ssim_scores, save_path):
    """Plot and save training losses and SSIM scores."""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot SSIM scores
    plt.subplot(1, 2, 2)
    plt.plot(ssim_scores, label='SSIM Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Score')
    plt.title('SSIM Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train(args):
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data loaders
    train_dataset = SketchFaceDataset(
        sketch_dir=args.train_sketch_dir,
        face_dir=args.train_photo_dir
    )
    test_dataset = SketchFaceDataset(
        sketch_dir=args.test_sketch_dir,
        face_dir=args.test_photo_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss functions and optimizers
    adv_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    optG = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Lists to store losses and metrics
    g_losses = []
    d_losses = []
    ssim_scores = []
    avg_ssim = 0

    # Training loop
    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_ssim = 0
        n_batches = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]")
        for sk, fc in loop:
            sk, fc = sk.to(device), fc.to(device)

            # Train Discriminator
            D_real = discriminator(fc)
            lossD_real = adv_loss(D_real, torch.ones_like(D_real, device=device))

            fake = generator(sk)
            D_fake = discriminator(fake.detach())
            lossD_fake = adv_loss(D_fake, torch.zeros_like(D_fake, device=device))

            lossD = 0.5 * (lossD_real + lossD_fake)
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # Train Generator
            D_fake2 = discriminator(fake)
            lossG_adv = adv_loss(D_fake2, torch.ones_like(D_fake2, device=device))
            lossG_l1 = l1_loss(fake, fc) * args.lambda_l1
            lossG = lossG_adv + lossG_l1

            optG.zero_grad()
            lossG.backward()
            optG.step()

            # Calculate SSIM
            with torch.no_grad():
                ssim_val = calculate_ssim(fake[0], fc[0])
                epoch_ssim += ssim_val

            # Update metrics
            epoch_g_loss += lossG.item()
            epoch_d_loss += lossD.item()
            n_batches += 1

            loop.set_postfix({
                'Loss D': f'{lossD.item():.4f}',
                'Loss G': f'{lossG.item():.4f}',
                'SSIM': f'{ssim_val:.4f}'
            })

        # Calculate epoch averages
        avg_g_loss = epoch_g_loss / n_batches
        avg_d_loss = epoch_d_loss / n_batches
        avg_ssim = epoch_ssim / n_batches

        # Store metrics
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        ssim_scores.append(avg_ssim)

        # Save model checkpoints and generate samples
        if (epoch + 1) % args.save_interval == 0:
            # Save checkpoints
            torch.save(generator.state_dict(), 
                      os.path.join(args.checkpoint_dir, f'generator_epoch_{epoch+1}.pth'))
            
            # Save test samples
            save_test_samples(generator, test_loader, device, epoch + 1, args.output_dir)
            
            # Plot and save losses
            plot_losses(g_losses, d_losses, ssim_scores,
                       os.path.join(args.output_dir, f'losses_epoch_{epoch+1}.png'))

    # Save final plots
    plot_losses(g_losses, d_losses, ssim_scores,
                os.path.join(args.output_dir, 'final_losses.png'))

    print(f"Training completed. Final average SSIM: {avg_ssim:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sketch to Face GAN Training')
    
    # Dataset parameters
    parser.add_argument('--train_sketch_dir', type=str, required=True, help='Training sketch directory')
    parser.add_argument('--train_photo_dir', type=str, required=True, help='Training photo directory')
    parser.add_argument('--test_sketch_dir', type=str, required=True, help='Testing sketch directory')
    parser.add_argument('--test_photo_dir', type=str, required=True, help='Testing photo directory')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100, help='L1 loss weight')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save samples and plots')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main() 