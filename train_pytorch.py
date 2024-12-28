import os
import gc
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp  # Für Mixed Precision
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

        # Wenn data 3D oder 4D ist, nimm den ersten 2D-Slice
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


def save_processed_images(processed_images, processed_file='processed_images.txt'):
    """Speichert die Namen der verarbeiteten Images."""
    with open(processed_file, 'w') as f:
        for image in processed_images:
            f.write(f"{image}\n")


def load_processed_images(processed_file='processed_images.txt'):
    """Lädt bereits verarbeitete Image-Filenames."""
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()


def prepare_dataset(input_folder, chunk_size=128, overlap=32, max_samples=30000):
    """Prepare the dataset using multiprocessing and save it as numpy arrays."""
    log(f"Preparing dataset from folder: {input_folder}")
    fits_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                  if f.lower().endswith(('.fits', '.fit'))]

    # Lade bereits verarbeitete Dateien
    processed_images = load_processed_images()

    # Filtere bereits verarbeitete aus
    new_files = [file for file in fits_files if file not in processed_images]
    log(f"Found {len(new_files)} new images to process.")

    # Falls keine neuen Dateien vorhanden sind, versuche vorhandene Arrays zu laden
    if not new_files:
        log("No new images found. Attempting to load existing dataset...")
        X_train = np.load('X_train.npy') if os.path.exists('X_train.npy') else None
        y_train = np.load('y_train.npy') if os.path.exists('y_train.npy') else None
        if X_train is None or y_train is None:
            log("No existing dataset found. Please provide FITS files in the input folder.")
            return np.array([]), np.array([])
        return X_train, y_train

    # Hier werden wir alle Chunks in X_train, y_train sammeln
    X_train, y_train = [], []

    # Batchweise multiprocessing
    batch_size = 8
    for i in range(0, len(new_files), batch_size):
        batch_files = new_files[i:i + batch_size]
        with Pool(processes=min(cpu_count(), len(batch_files))) as pool:
            results = pool.starmap(
                process_file,
                [(file_path, chunk_size, overlap) for file_path in batch_files]
            )

        # Sammle Ergebnisse
        for result in results:
            for chunk, background in result:
                # Original
                X_train.append(chunk)
                y_train.append(background)
                # Horizontal gespiegelt
                X_train.append(np.fliplr(chunk))
                y_train.append(np.fliplr(background))
                # 90° rotiert
                X_train.append(np.rot90(chunk))
                y_train.append(np.rot90(background))

                # Begrenze Anzahl Samples
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
    """Speichere jeden Chunk als einzelne .npy Datei."""
    os.makedirs(chunk_dir, exist_ok=True)
    for i, (chunk, label) in enumerate(zip(chunks, labels)):
        np.save(f"{chunk_dir}/chunk_{i}.npy", chunk)
        np.save(f"{chunk_dir}/label_{i}.npy", label)
    return i + 1  # Anzahl gespeicherter Chunks


def load_chunk_batch(chunk_dir, start_idx, batch_size):
    """Lade batch_size Chunks von Disk."""
    X_batch = []
    y_batch = []
    for i in range(start_idx, min(start_idx + batch_size, len(os.listdir(chunk_dir)) // 2)):
        try:
            X_batch.append(np.load(f"{chunk_dir}/chunk_{i}.npy"))
            y_batch.append(np.load(f"{chunk_dir}/label_{i}.npy"))
        except:
            break
    return np.array(X_batch), np.array(y_batch)


class ChunkDataset(Dataset):
    def __init__(self, chunk_dir, is_training=True):
        """
        chunk_dir: Verzeichnis mit chunk_i.npy und label_i.npy Dateien.
        is_training: Ob das Dataset für das Training (z.B. mit Augmentation) verwendet wird.
        """
        self.chunk_dir = chunk_dir
        self.is_training = is_training

        all_files = os.listdir(self.chunk_dir)
        # Jede Chunk/Label-Paar-Datei => 2 Dateien: chunk_i.npy und label_i.npy
        # Anzahl Chunks = len(all_files) // 2
        self.n_chunks = len(all_files) // 2

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        """
        Lädt chunk_{idx}.npy und label_{idx}.npy, gibt (chunk, label) zurück.
        """
        X = np.load(os.path.join(self.chunk_dir, f"chunk_{idx}.npy"))
        y = np.load(os.path.join(self.chunk_dir, f"label_{idx}.npy"))

        # Falls gewünscht, könnte man hier weitere PyTorch-Augmentationen machen
        # (z.B. Random Rotate, Random Flip), aber das haben wir oben schon gemacht.

        # Konvertierung in PyTorch-Tensoren
        # Shape [H, W, 1] => [1, H, W] (typische PyTorch-Konvention: [C, H, W])
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

        # Decoder
        self.upconv1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.upconv0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        c1 = self.enc_conv0(x)  # [B, 32, H, W]
        p1 = self.pool0(c1)  # [B, 32, H/2, W/2]

        c2 = self.enc_conv1(p1)  # [B, 64, H/2, W/2]
        p2 = self.pool1(c2)  # [B, 64, H/4, W/4]

        # Bottleneck
        c3 = self.bottleneck(p2)  # [B, 128, H/4, W/4]

        # Decoder
        u4 = self.upconv1(c3)  # [B, 128, H/2, W/2]
        cat4 = torch.cat([u4, c2], dim=1)  # [B, 128+64, H/2, W/2]
        c4 = self.dec_conv1(cat4)  # [B, 64, H/2, W/2]

        u5 = self.upconv0(c4)  # [B, 64, H, W]
        cat5 = torch.cat([u5, c1], dim=1)  # [B, 64+32, H, W]
        c5 = self.dec_conv0(cat5)  # [B, 32, H, W]

        out = self.out_conv(c5)  # [B, 1, H, W]
        return out

def create_dataloader(chunk_dir, batch_size, is_training=True):
    dataset = ChunkDataset(chunk_dir, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=2)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-4,
    model_checkpoint_path='chunk_based_unet.pt',
    patience=20
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # For Mixed Precision
    scaler = amp.GradScaler()

    # Early Stopping & LR Plateaus
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training Loop
        train_bar = tqdm(train_loader, desc="Training", unit="batch")
        for batch_idx, (inputs, targets) in enumerate(train_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            # Mixed Precision
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({"Loss": loss.item()})

        train_loss = running_loss / len(train_loader.dataset)
        log(f"Train Loss: {train_loss:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc="Validating", unit="batch")
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_bar.set_postfix({"Loss": loss.item()})

        val_loss /= len(val_loader.dataset)
        log(f"Val Loss: {val_loss:.4f}")

        # Scheduler Adjustment
        scheduler.step(val_loss)

        # Early Stopping / Checkpoint
        if val_loss < best_val_loss - 1e-4:  # min_delta
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save(model.state_dict(), model_checkpoint_path)
            log("Model checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Memory Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Load best model
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path))
    return model


if __name__ == '__main__':
    input_folder = "input"
    chunk_size = 128
    overlap = 32
    batch_size = 128
    epochs = 50
    model_checkpoint_path = 'chunk_based_unet.pt'
    chunk_dir = 'chunks'
    val_chunk_dir = 'val_chunks'

    # 1. Falls keine Chunks existieren: Datensatz verarbeiten
    if not os.path.exists(chunk_dir):
        log("Processing dataset and saving chunks...")
        X_train, y_train = prepare_dataset(input_folder, chunk_size, overlap, max_samples=30000)

        if len(X_train) == 0:
            log("No training data available. Exiting.")
            exit(0)

        # Split in Trainings- und Validierungsdaten
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Chunks auf Platte speichern
        n_train = save_chunks(X_train, y_train, chunk_dir)
        n_val = save_chunks(X_val, y_val, val_chunk_dir)
        log(f"Saved {n_train} training chunks and {n_val} validation chunks")

        # Speicher freigeben
        del X_train, X_val, y_train, y_val
        gc.collect()

    # 2. DataLoader anlegen
    train_loader = create_dataloader(chunk_dir, batch_size, is_training=True)
    val_loader = create_dataloader(val_chunk_dir, batch_size, is_training=False)

    # 3. Model laden oder neu erstellen
    model = UNet()
    if os.path.exists(model_checkpoint_path):
        log(f"Restoring model from {model_checkpoint_path}...")
        model.load_state_dict(torch.load(model_checkpoint_path))
    else:
        log("Creating new model...")

    # 4. Training
    log("Starting training...")
    try:
        model = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=1e-4,
            model_checkpoint_path=model_checkpoint_path,
            patience=20
        )
    except Exception as e:
        log(f"Training failed with error: {str(e)}")
        torch.save(model.state_dict(), 'emergency_backup_model.pt')
        raise e

    log("Training finished successfully.")