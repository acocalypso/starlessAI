import os
import gc
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def log(message):
    """Log messages to console."""
    print(f"[INFO] {message}")


def load_fits_image(file_path):
    """Load a FITS image and ensure it's 2D."""
    with fits.open(file_path) as hdul:
        data = hdul[0].data.astype(np.float32)

        # If data is 3D or 4D, take first 2D slice
        if data.ndim > 2:
            data = data[0] if data.ndim == 3 else data[0][0]
            log(f"Reduced {file_path} dimensions to 2D")

        if data.ndim != 2:
            raise ValueError(f"Could not convert {file_path} to 2D array. Shape: {data.shape}")

        return data


def divide_into_chunks(image, chunk_size=128, overlap=32):
    """Divide an image into chunks with overlap."""
    h, w = image.shape
    step = chunk_size - overlap
    chunks = []

    for i in range(0, h - overlap, step):
        for j in range(0, w - overlap, step):
            chunk = image[i:i + chunk_size, j:j + chunk_size]
            if chunk.shape == (chunk_size, chunk_size):
                chunks.append((chunk, i, j))

    return chunks


def simulate_background(chunk):
    """Simulate background by applying Gaussian blur."""
    return gaussian_filter(chunk, sigma=10)


def process_file(file_path, chunk_size=128, overlap=32):
    """Load a FITS file, divide it into chunks, and generate training pairs."""
    log(f"Processing file: {file_path}")
    image = load_fits_image(file_path)
    chunks = divide_into_chunks(image, chunk_size, overlap)

    data_pairs = []
    for chunk, _i, _j in chunks:
        background = simulate_background(chunk)
        data_pairs.append((chunk, background))

    return data_pairs


class ChunkDataset(Dataset):
    def __init__(self, chunk_dir, is_training=True):
        self.chunk_dir = chunk_dir
        self.is_training = is_training
        all_files = os.listdir(self.chunk_dir)
        self.n_chunks = len(all_files) // 2

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        X = np.load(os.path.join(self.chunk_dir, f"chunk_{idx}.npy"))
        y = np.load(os.path.join(self.chunk_dir, f"label_{idx}.npy"))
        
        X = torch.tensor(X.transpose(2, 0, 1), dtype=torch.float32)
        y = torch.tensor(y.transpose(2, 0, 1), dtype=torch.float32)

        return X, y


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pool0 = nn.MaxPool2d(2)

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pool1 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder for starless image
        self.upconv1_starless = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv1_starless = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.upconv0_starless = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv0_starless = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.out_conv_starless = nn.Conv2d(32, 1, 1)

        # Decoder for stars
        self.upconv1_stars = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv1_stars = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.upconv0_stars = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv0_stars = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.out_conv_stars = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        c1 = self.enc_conv0(x)
        p1 = self.pool0(c1)

        c2 = self.enc_conv1(p1)
        p2 = self.pool1(c2)

        # Bottleneck
        c3 = self.bottleneck(p2)

        # Decoder for starless image
        u4_starless = self.upconv1_starless(c3)
        cat4_starless = torch.cat([u4_starless, c2], dim=1)
        c4_starless = self.dec_conv1_starless(cat4_starless)

        u5_starless = self.upconv0_starless(c4_starless)
        cat5_starless = torch.cat([u5_starless, c1], dim=1)
        c5_starless = self.dec_conv0_starless(cat5_starless)

        out_starless = self.out_conv_starless(c5_starless)

        # Decoder for stars
        u4_stars = self.upconv1_stars(c3)
        cat4_stars = torch.cat([u4_stars, c2], dim=1)
        c4_stars = self.dec_conv1_stars(cat4_stars)

        u5_stars = self.upconv0_stars(c4_stars)
        cat5_stars = torch.cat([u5_stars, c1], dim=1)
        c5_stars = self.dec_conv0_stars(cat5_stars)

        out_stars = self.out_conv_stars(c5_stars)

        return out_starless, out_stars


def load_model_with_conversion(model, checkpoint_path, device):
    """Load a model checkpoint with state dict conversion for compatibility"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    
    # Convert old state dict keys to new format
    for key, value in state_dict.items():
        if 'dec_conv1.' in key:
            new_key = key.replace('dec_conv1.', 'dec_conv1_starless.')
            new_state_dict[new_key] = value
            stars_key = key.replace('dec_conv1.', 'dec_conv1_stars.')
            new_state_dict[stars_key] = value.clone()
        elif 'dec_conv0.' in key:
            new_key = key.replace('dec_conv0.', 'dec_conv0_starless.')
            new_state_dict[new_key] = value
            stars_key = key.replace('dec_conv0.', 'dec_conv0_stars.')
            new_state_dict[stars_key] = value.clone()
        elif 'out_conv.' in key:
            new_key = key.replace('out_conv.', 'out_conv_starless.')
            new_state_dict[new_key] = value
            stars_key = key.replace('out_conv.', 'out_conv_stars.')
            new_state_dict[stars_key] = value.clone()
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    return model


def create_dataloader(chunk_dir, batch_size, is_training=True):
    dataset = ChunkDataset(chunk_dir, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=2)


def process_image(model, image_path, device, chunk_size=128, overlap=32):
    """Process a single image to remove stars"""
    # Load and preprocess image
    image = load_fits_image(image_path)
    chunks = divide_into_chunks(image, chunk_size, overlap)
    
    # Prepare output arrays
    h, w = image.shape
    starless_output = np.zeros_like(image)
    stars_output = np.zeros_like(image)
    weights = np.zeros_like(image)
    
    model.eval()
    with torch.no_grad():
        for chunk, i, j in tqdm(chunks, desc="Processing chunks"):
            # Prepare input
            chunk_tensor = torch.tensor(chunk[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
            chunk_tensor = chunk_tensor.to(device)
            
            # Process chunk
            starless_chunk, stars_chunk = model(chunk_tensor)
            
            # Convert back to numpy
            starless_chunk = starless_chunk.cpu().numpy()[0, 0]
            stars_chunk = stars_chunk.cpu().numpy()[0, 0]
            
            # Create weight mask (higher weights in the center)
            y, x = np.ogrid[:chunk_size, :chunk_size]
            weight = (1 - ((x - chunk_size/2)**2 + (y - chunk_size/2)**2) / ((chunk_size/2)**2))
            weight = np.clip(weight, 0, 1)
            
            # Add to output with weights
            starless_output[i:i+chunk_size, j:j+chunk_size] += starless_chunk * weight
            
            # For stars, only keep pixels where the star mask is significant
            star_threshold = 0.1  # Adjust this threshold as needed
            star_mask = stars_chunk > star_threshold
            original_stars = chunk * star_mask
            stars_output[i:i+chunk_size, j:j+chunk_size] = np.maximum(
                stars_output[i:i+chunk_size, j:j+chunk_size],
                original_stars
            )
            weights[i:i+chunk_size, j:j+chunk_size] += weight
    
    # Normalize by weights (only for starless output)
    mask = weights > 0
    starless_output[mask] /= weights[mask]
    
    # For stars output, use original pixel values where stars were detected
    final_stars_output = np.zeros_like(image)
    stars_mask = stars_output > 0
    final_stars_output[stars_mask] = image[stars_mask]
    
    return starless_output, final_stars_output


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    model_path = "chunk_based_unet.pt"
    input_dir = "removeStars"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and load model
    model = UNet().to(device)
    model = load_model_with_conversion(model, model_path, device)
    
    # Process all FITS files in input directory
    fits_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.fits', '.fit'))]
    
    for fits_file in fits_files:
        input_path = os.path.join(input_dir, fits_file)
        base_name = os.path.splitext(fits_file)[0]
        
        try:
            starless_img, stars_img = process_image(model, input_path, device)
            
            # Save outputs
            starless_path = os.path.join(output_dir, f"{base_name}_starless.fits")
            stars_path = os.path.join(output_dir, f"{base_name}_stars.fits")
            
            fits.writeto(starless_path, starless_img, overwrite=True)
            fits.writeto(stars_path, stars_img, overwrite=True)
            
            log(f"Successfully processed {fits_file}")
            
        except Exception as e:
            log(f"Error processing {fits_file}: {str(e)}")
            continue


if __name__ == "__main__":
    main()