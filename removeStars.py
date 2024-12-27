import os
import numpy as np
from astropy.io import fits
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import gc

def log(message):
    """Log messages to console."""
    print(f"[INFO] {message}")

def load_image(file_path):
    """Load image from various formats (FITS, TIFF, XISF)."""
    file_ext = file_path.lower().split('.')[-1]
    
    if file_ext in ['fits', 'fit']:
        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim > 2:
                data = data[0] if data.ndim == 3 else data[0][0]
    elif file_ext in ['tiff', 'tif']:
        data = np.array(Image.open(file_path)).astype(np.float32)
        if data.ndim > 2:
            data = np.mean(data, axis=2)  # Convert RGB to grayscale
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return data

def normalize_image(image):
    """Normalize image to 0-1 range."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image - min_val
    return (image - min_val) / (max_val - min_val), min_val, max_val

def denormalize_image(image, min_val, max_val):
    """Denormalize image back to original range."""
    return image * (max_val - min_val) + min_val

def process_image(model, image, chunk_size=128, overlap=32):
    """Process large image by dividing into overlapping chunks."""
    h, w = image.shape
    pad_h = (chunk_size - h % chunk_size) if h % chunk_size != 0 else 0
    pad_w = (chunk_size - w % chunk_size) if w % chunk_size != 0 else 0
    
    # Pad image
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    
    # Create output arrays
    background = np.zeros_like(padded)
    weight_map = np.zeros_like(padded)
    
    # Process chunks with overlap
    for i in range(0, ph - chunk_size + 1, chunk_size - overlap):
        for j in range(0, pw - chunk_size + 1, chunk_size - overlap):
            # Extract chunk
            chunk = padded[i:i+chunk_size, j:j+chunk_size]
            
            # Create weight mask for blending
            weight = np.ones((chunk_size, chunk_size))
            if overlap > 0:
                weight[:overlap, :] *= np.linspace(0, 1, overlap)[:, np.newaxis]
                weight[-overlap:, :] *= np.linspace(1, 0, overlap)[:, np.newaxis]
                weight[:, :overlap] *= np.linspace(0, 1, overlap)
                weight[:, -overlap:] *= np.linspace(1, 0, overlap)
            
            # Process chunk
            processed = model.predict(chunk[np.newaxis, ..., np.newaxis], verbose=0)[0, ..., 0]
            
            # Add to output with weight
            background[i:i+chunk_size, j:j+chunk_size] += processed * weight
            weight_map[i:i+chunk_size, j:j+chunk_size] += weight
    
    # Average overlapping regions
    background = background / np.maximum(weight_map, 1e-10)
    
    # Remove padding
    background = background[:h, :w]
    
    return background

def save_image(image, output_path):
    """Save image in the same format as input."""
    file_ext = output_path.lower().split('.')[-1]
    
    if file_ext in ['fits', 'fit']:
        hdu = fits.PrimaryHDU(image.astype(np.float32))
        hdu.writeto(output_path, overwrite=True)
    elif file_ext in ['tiff', 'tif']:
        # Scale to 16-bit range
        scaled = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 65535).astype(np.uint16)
        Image.fromarray(scaled).save(output_path)

def process_directory(model_path, input_dir, output_dir):
    """Process all images in input directory."""
    # Create output directories
    starless_dir = os.path.join(output_dir, 'starless')
    stars_dir = os.path.join(output_dir, 'stars_only')
    os.makedirs(starless_dir, exist_ok=True)
    os.makedirs(stars_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    log(f"Loaded model from {model_path}")
    
    # Get list of supported files
    supported_extensions = ('.fits', '.fit', '.tiff', '.tif', '.xisf')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]
    
    for file in files:
        try:
            input_path = os.path.join(input_dir, file)
            base_name = os.path.splitext(file)[0]
            output_ext = os.path.splitext(file)[1]
            
            log(f"Processing {file}...")
            
            # Load and normalize image
            image = load_image(input_path)
            normalized, min_val, max_val = normalize_image(image)
            
            # Process image
            background = process_image(model, normalized)
            
            # Calculate stars by subtracting background
            stars = normalized - background
            
            # Denormalize images
            background = denormalize_image(background, min_val, max_val)
            stars = denormalize_image(stars, min_val, max_val)
            
            # Save outputs
            save_image(background, os.path.join(starless_dir, f"{base_name}_starless{output_ext}"))
            save_image(stars, os.path.join(stars_dir, f"{base_name}_stars{output_ext}"))
            
            log(f"Saved processed images for {file}")
            
            # Clean up memory
            gc.collect()
            
        except Exception as e:
            log(f"Error processing {file}: {str(e)}")
            continue

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "chunk_based_unet.keras"  # Your trained model path
    INPUT_DIR = "removeStars"              # Input directory
    OUTPUT_DIR = "output"                  # Output directory
    
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            log(f"GPU(s) found and configured: {physical_devices}")
        except RuntimeError as e:
            log(f"GPU configuration error: {e}")
    else:
        log("No GPU detected, using CPU")
    
    # Process images
    process_directory(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)