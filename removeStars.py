import os
import numpy as np
from astropy.io import fits
from tensorflow.keras.models import load_model

def load_fits_image(file_path):
    """Load a FITS image and return data and header."""
    with fits.open(file_path) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data.astype(np.float32)
        
        # Handle different dimensional data
        if len(data.shape) > 2:
            print(f"[WARNING] Input image has {len(data.shape)} dimensions: {data.shape}")
            # If 3D or 4D, take the first frame/channel
            while len(data.shape) > 2:
                data = data[0]
            print("[INFO] Converted to 2D image with shape:", data.shape)
        
        print(f"[DEBUG] Input image stats - Min: {np.min(data):.2f}, Max: {np.max(data):.2f}, Mean: {np.mean(data):.2f}")
        print(f"[DEBUG] Input image dtype: {data.dtype}")
        return data, header

def pad_image(image, chunk_size, overlap):
    """Pad image to ensure it can be divided into full chunks."""
    h, w = image.shape
    step = chunk_size - overlap
    
    # Calculate needed padding
    pad_h = (step - h % step) % step if h % step != 0 else 0
    pad_w = (step - w % step) % step if w % step != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        print(f"[INFO] Padding image with {pad_h} rows and {pad_w} columns")
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        return padded, (pad_h, pad_w)
    return image, (0, 0)

def divide_into_chunks(image, chunk_size=128, overlap=32):
    """Divide an image into chunks with overlap."""
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    h, w = image.shape
    step = chunk_size - overlap
    chunks = []
    
    # Pad image if needed
    padded_image, (pad_h, pad_w) = pad_image(image, chunk_size, overlap)
    padded_h, padded_w = padded_image.shape
    
    for i in range(0, padded_h - chunk_size + 1, step):
        for j in range(0, padded_w - chunk_size + 1, step):
            chunk = padded_image[i:i+chunk_size, j:j+chunk_size]
            chunks.append((chunk, i, j))
    
    print(f"[DEBUG] Created {len(chunks)} chunks of size {chunk_size}x{chunk_size}")
    return chunks, (pad_h, pad_w)

def reconstruct_image(chunks, predictions, scaling_factors, chunk_size, overlap, original_shape, padding):
    """Reconstruct the full image from predicted chunks."""
    pad_h, pad_w = padding
    h, w = original_shape
    padded_h = h + pad_h
    padded_w = w + pad_w
    
    step = chunk_size - overlap
    output = np.zeros((padded_h, padded_w))
    count = np.zeros((padded_h, padded_w))
    
    for (_, i, j), prediction, (chunk_min, chunk_max) in zip(chunks, predictions, scaling_factors):
        # Ensure we're working with 2D data
        pred = prediction.squeeze()
        
        # Denormalize the prediction
        denorm_pred = denormalize_prediction(pred, chunk_min, chunk_max)
        
        # Add to the output and count arrays
        output[i:i+chunk_size, j:j+chunk_size] += denorm_pred
        count[i:i+chunk_size, j:j+chunk_size] += 1
    
    # Average overlapping regions
    mask = count > 0
    output[mask] /= count[mask]
    
    # Remove padding to get back to original size
    output = output[:h, :w]
    
    print(f"[DEBUG] Reconstructed image stats - Min: {np.min(output):.2f}, Max: {np.max(output):.2f}, Mean: {np.mean(output):.2f}")
    return output

def normalize_chunk(chunk):
    """Normalize chunk to [0,1] range and return scaling factors."""
    chunk_min = np.min(chunk)
    chunk_max = np.max(chunk)
    if chunk_max == chunk_min:
        return np.zeros_like(chunk), chunk_min, chunk_max
    normalized = (chunk - chunk_min) / (chunk_max - chunk_min)
    return normalized, chunk_min, chunk_max

def denormalize_prediction(pred, original_min, original_max):
    """Denormalize prediction back to original scale."""
    return pred * (original_max - original_min) + original_min

def save_fits_image(data, header, output_file, image_type):
    """Save FITS image with proper header preservation."""
    # Ensure data is in the correct range and type
    if image_type == 'starless':  # Only clip starless image
        data = np.clip(data, 0, None)
    
    # Convert to the same data type as input if it was integer
    if header.get('BITPIX', 0) > 0:
        data = np.round(data).astype(np.uint16)
    
    # Update header
    header['DATAMIN'] = np.min(data)
    header['DATAMAX'] = np.max(data)
    header['HISTORY'] = f'Processed with star removal - {image_type} image'
    
    # Create HDU and save
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(output_file, overwrite=True)
    
    print(f"[DEBUG] {image_type} image saved stats - "
          f"Min: {np.min(data):.2f}, Max: {np.max(data):.2f}, Mean: {np.mean(data):.2f}")

# Main processing
try:
    # Load the model
    model = load_model("chunk_based_unet.keras")
    print("[INFO] Model loaded successfully.")
    print(f"[DEBUG] Model input shape: {model.input_shape}")
    print(f"[DEBUG] Model output shape: {model.output_shape}")

    # Configuration
    remove_stars_folder = "removeStars"
    output_folder = "output"
    chunk_size = 128
    overlap = 32

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each FITS file
    for filename in os.listdir(remove_stars_folder):
        if filename.endswith(".fits"):
            input_file = os.path.join(remove_stars_folder, filename)
            print(f"\n[INFO] Processing {input_file}...")

            try:
                # Load image
                image, header = load_fits_image(input_file)
                
                # Divide into chunks
                chunks, padding = divide_into_chunks(image, chunk_size, overlap)
                predictions = []
                scaling_factors = []
                
                # Process each chunk
                for chunk, _, _ in chunks:
                    # Normalize chunk
                    normalized_chunk, chunk_min, chunk_max = normalize_chunk(chunk)
                    scaling_factors.append((chunk_min, chunk_max))
                    
                    # Predict
                    input_chunk = normalized_chunk[np.newaxis, ..., np.newaxis]
                    pred = model.predict(input_chunk, verbose=0)
                    predictions.append(pred)
                    
                    # Debug first few chunks
                    if len(predictions) <= 3:
                        print(f"[DEBUG] Chunk {len(predictions)} prediction stats - "
                              f"Min: {np.min(pred):.2f}, Max: {np.max(pred):.2f}, Mean: {np.mean(pred):.2f}")
                        print(f"[DEBUG] Chunk {len(predictions)} scale - "
                              f"Min: {chunk_min:.2f}, Max: {chunk_max:.2f}")

                # Reconstruct full image
                starless_image = reconstruct_image(chunks, predictions, scaling_factors, 
                                                 chunk_size, overlap, image.shape, padding)
                
                # Calculate stars-only image
                stars_only_image = image - starless_image

                # Save outputs
                starless_output = os.path.join(output_folder, f"starless_{filename}")
                stars_output = os.path.join(output_folder, f"stars_{filename}")
                
                save_fits_image(starless_image, header.copy(), starless_output, "starless")
                save_fits_image(stars_only_image, header.copy(), stars_output, "stars")

            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {str(e)}")
                continue

except Exception as e:
    print(f"[ERROR] An error occurred: {str(e)}")
    raise