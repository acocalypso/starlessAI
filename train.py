import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from tensorflow.keras.models import load_model
import gc

# Shortcuts for convenience
keras = tf.keras
layers = keras.layers
models = keras.models

# GPU Configuration with more conservative memory settings
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            # More conservative memory limit (3GB)
            tf.config.set_logical_device_configuration(
                device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3)]  # 3GB limit
            )
        print(f"GPU(s) found and configured: {physical_devices}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected, using CPU")

# Enable mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
print("Mixed precision training enabled with policy: mixed_float16.")

# =============================
# 1. Helper Functions
# =============================

def log(message):
    """Log messages to console."""
    print(f"[INFO] {message}")

def load_fits_image(file_path):
    """Load a FITS image and ensure it's 2D."""
    with fits.open(file_path) as hdul:
        # Get the primary HDU data
        data = hdul[0].data.astype(np.float32)
        
        # If data is 3D or 4D, take the first 2D slice
        if data.ndim > 2:
            data = data[0] if data.ndim == 3 else data[0][0]
            log(f"Reduced {file_path} dimensions to 2D")
            
        # Ensure we have a 2D array
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
            chunk = image[i:i+chunk_size, j:j+chunk_size]
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

def save_processed_images(processed_images, processed_file='processed_images.txt'):
    """Save list of processed image filenames."""
    with open(processed_file, 'w') as f:
        for image in processed_images:
            f.write(f"{image}\n")

def load_processed_images(processed_file='processed_images.txt'):
    """Load list of processed image filenames."""
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def prepare_dataset(input_folder, chunk_size=128, overlap=32, max_samples=10000):
    """Prepare the dataset using multiprocessing with memory constraints."""
    log(f"Preparing dataset from folder: {input_folder}")
    print(os.listdir("input"))
    fits_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.fits', '.fit'))]
    
    # Get previously processed images
    processed_images = load_processed_images()
    
    # Filter out already processed files
    new_files = [file for file in fits_files if file not in processed_images]
    log(f"Found {len(new_files)} new images to process.")

    if not new_files:
        log("No new images found. Attempting to load existing dataset...")
        X_train = np.load('X_train.npy') if os.path.exists('X_train.npy') else None
        y_train = np.load('y_train.npy') if os.path.exists('y_train.npy') else None
        if X_train is None or y_train is None:
            log("No existing dataset found. Please provide FITS files in the input folder.")
            return np.array([]), np.array([])
        return X_train, y_train

    # Process files in larger batches
    batch_size = 8  # Increased batch size
    X_train, y_train = [], []
    
    for i in range(0, len(new_files), batch_size):
        batch_files = new_files[i:i + batch_size]
        with Pool(processes=min(cpu_count(), len(batch_files))) as pool:
            results = pool.starmap(process_file, [(file_path, chunk_size, overlap) for file_path in batch_files])
        
        # Process results
        for result in results:
            for chunk, background in result:
                # Add original data
                X_train.append(chunk)
                y_train.append(background)
                
                # Add flipped version
                X_train.append(np.fliplr(chunk))
                y_train.append(np.fliplr(background))
                
                # Add rotated version
                X_train.append(np.rot90(chunk))
                y_train.append(np.rot90(background))
                
                # Limit total samples
                if len(X_train) >= max_samples:
                    log(f"Reached maximum sample limit of {max_samples}")
                    processed_images.update(batch_files)
                    save_processed_images(processed_images)
                    return np.array(X_train)[..., np.newaxis], np.array(y_train)[..., np.newaxis]
        
        log(f"Processed {len(X_train)} samples so far...")
        processed_images.update(batch_files)
        save_processed_images(processed_images)
        gc.collect()
    
    return np.array(X_train)[..., np.newaxis], np.array(y_train)[..., np.newaxis]

def save_chunks(chunks, labels, chunk_dir='chunks'):
    """Save individual chunks to disk."""
    os.makedirs(chunk_dir, exist_ok=True)
    for i, (chunk, label) in enumerate(zip(chunks, labels)):
        np.save(f"{chunk_dir}/chunk_{i}.npy", chunk)
        np.save(f"{chunk_dir}/label_{i}.npy", label)
    return i + 1  # Return number of chunks saved

def load_chunk_batch(chunk_dir, start_idx, batch_size):
    """Load a batch of chunks from disk."""
    X_batch = []
    y_batch = []
    for i in range(start_idx, min(start_idx + batch_size, len(os.listdir(chunk_dir))//2)):
        try:
            X_batch.append(np.load(f"{chunk_dir}/chunk_{i}.npy"))
            y_batch.append(np.load(f"{chunk_dir}/label_{i}.npy"))
        except:
            break
    return np.array(X_batch), np.array(y_batch)

# =============================
# 2. Memory-Optimized Dataset Generator
# =============================

class ChunkDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, chunk_dir, batch_size, is_training=True):
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.is_training = is_training
        self.n_chunks = len(os.listdir(chunk_dir))//2
        self.indexes = np.arange(self.n_chunks)
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(self.n_chunks / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        
        for i in batch_indexes:
            X = np.load(f"{self.chunk_dir}/chunk_{i}.npy")
            y = np.load(f"{self.chunk_dir}/label_{i}.npy")
            X_batch.append(X)
            y_batch.append(y)
            
        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

# =============================
# 3. Memory Cleanup Callback
# =============================

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

# =============================
# 4. Main Code
# =============================

if __name__ == '__main__':
    # Define parameters
    input_folder = "input"
    chunk_size = 128
    overlap = 32
    batch_size = 16  # Increased batch size
    epochs = 50     # Increased epochs
    model_checkpoint_path = 'chunk_based_unet.keras'
    chunk_dir = 'chunks'
    val_chunk_dir = 'val_chunks'

    # Process and save chunks to disk
    if not os.path.exists(chunk_dir):
        log("Processing dataset and saving chunks...")
        X_train, y_train = prepare_dataset(input_folder, chunk_size, overlap, max_samples=10000)
        
        # Split dataset
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Save chunks
        n_train = save_chunks(X_train, y_train, chunk_dir)
        n_val = save_chunks(X_val, y_val, val_chunk_dir)
        log(f"Saved {n_train} training chunks and {n_val} validation chunks")
        
        # Clear memory
        del X_train, X_val, y_train, y_val
        gc.collect()

    # Create data generators
    train_generator = ChunkDataGenerator(chunk_dir, batch_size, is_training=True)
    val_generator = ChunkDataGenerator(val_chunk_dir, batch_size, is_training=False)

    # Load or create model
    if os.path.exists(model_checkpoint_path):
        log(f"Restoring model from {model_checkpoint_path}...")
        model = load_model(model_checkpoint_path)
    else:
        # Build a new model
        log("Creating new model...")
        print(os.listdir("input"))
        def unet_model(input_shape):
            inputs = layers.Input(shape=input_shape)

            # Encoder
            c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
            c1 = layers.Dropout(0.1)(c1)
            p1 = layers.MaxPooling2D((2, 2))(c1)

            c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
            c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
            c2 = layers.Dropout(0.1)(c2)
            p2 = layers.MaxPooling2D((2, 2))(c2)

            # Bottleneck
            c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
            c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
            c3 = layers.Dropout(0.2)(c3)

            # Decoder
            u4 = layers.UpSampling2D((2, 2))(c3)
            u4 = layers.concatenate([u4, c2])
            c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
            c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
            c4 = layers.Dropout(0.1)(c4)

            u5 = layers.UpSampling2D((2, 2))(c4)
            u5 = layers.concatenate([u5, c1])
            c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
            c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
            c5 = layers.Dropout(0.1)(c5)
            
            outputs = layers.Conv2D(1, (1, 1), activation='linear')(c5)
            return models.Model(inputs=[inputs], outputs=[outputs])

        model = unet_model(input_shape=(chunk_size, chunk_size, 1))
    
    # Use mixed precision optimizer
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=0.0001)
    )

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_checkpoint_path,
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=20,        # Increased patience
            monitor='val_loss',
            restore_best_weights=True,
            min_delta=0.0001   # Added minimum delta for improvement
        ),
        tf.keras.callbacks.ReduceLROnPlateau(  # Added learning rate reduction
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        MemoryCleanupCallback()
    ]

    # Train model
    log("Starting training...")
    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        log(f"Training failed with error: {str(e)}")
        model.save('emergency_backup_model.keras')
        raise e