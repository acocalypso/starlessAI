import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError
from astropy.io import fits
from tqdm import tqdm
import cv2
import random
import matplotlib.pyplot as plt
import glob
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications import VGG16

# Konfiguration
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
DATA_AUGMENTATION = True
MIXED_PRECISION = True
MODEL_PATH = "models/starless_model"
TRAINING_DATA_PATH = "data/training"
VALIDATION_DATA_PATH = "data/validation"
TEST_DATA_PATH = "data/test"
USE_MASKS = True  # Setzt auf True, wenn Sternmasken verwendet werden sollen

def load_image(file_path, normalize=True):
    """Lädt ein Bild im FITS- oder PNG-Format."""
    try:
        if file_path.endswith('.fits'):
            # FITS-Datei laden
            with fits.open(file_path) as hdul:
                image = hdul[0].data
                
                # Wenn Bild 3-dimensional ist, nehme das erste Bild
                if len(image.shape) > 2:
                    image = image[0]
                    
                # Negative Werte auf 0 setzen
                image = np.maximum(image, 0)
        else:
            # PNG oder andere Bildformate
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            # Alpha-Kanal entfernen, falls vorhanden
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = image[:,:,:3]
                
            # In Graustufen konvertieren, falls RGB
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if normalize:
            # Für PNG-Bilder normalisieren
            if file_path.endswith(('.png', '.jpg', '.jpeg')):
                image = image.astype(np.float32) / 255.0
            else:
                # Für FITS-Dateien logarithmische Normalisierung
                image = np.log1p(image)
                min_val = np.min(image)
                max_val = np.max(image)
                if max_val > min_val:
                    image = (image - min_val) / (max_val - min_val)
        
        return image
            
    except Exception as e:
        print(f"Fehler beim Laden von {file_path}: {e}")
        return None

def apply_mask_to_remove_stars(image, mask):
    """Wendet eine Sternmaske auf ein Bild an, um Sterne zu entfernen."""
    # Konvertieren der Maske in ein binäres Format (0, 1)
    binary_mask = mask > 0.5
    
    # Inpainted-Version erstellen (Bereiche unter der Maske werden interpoliert)
    inpainted = cv2.inpaint(
        (image * 255).astype(np.uint8),
        binary_mask.astype(np.uint8) * 255,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )
    
    # Zurück zu float32 normalisieren
    inpainted = inpainted.astype(np.float32) / 255.0
    
    return inpainted

def simulate_spikes(image, max_spikes=4, max_length=30, max_width=3, intensity=1.0):
    """Simuliert realistische Diffraktionsspikes bei Sternen."""
    result = image.copy()
    height, width = image.shape[:2]
    
    # Finde helle Pixel (potentielle Sterne)
    threshold = np.percentile(image, 99)
    bright_points = np.where(image > threshold)
    
    # Zufällig einige helle Punkte auswählen
    if len(bright_points[0]) > 0:
        num_points = min(20, len(bright_points[0]))
        indices = np.random.choice(len(bright_points[0]), num_points, replace=False)
        
        for i in indices:
            y, x = bright_points[0][i], bright_points[1][i]
            brightness = image[y, x]
            num_spikes = random.randint(2, max_spikes)
            
            # Spikes an diesem Punkt erstellen
            for _ in range(num_spikes):
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(10, max_length)
                width = random.randint(1, max_width)
                
                # Spike-Endpunkte berechnen
                end_x = int(x + length * np.cos(angle))
                end_y = int(y + length * np.sin(angle))
                
                # Spike zeichnen
                cv2.line(result, (x, y), (end_x, end_y), float(brightness * intensity), width)
    
    return result

def simulate_coma(image, strength=0.5):
    """Simuliert Coma-Aberration, besonders am Bildrand."""
    height, width = image.shape[:2]
    result = image.copy()
    
    # Erstelle eine Maske, die zum Rand hin stärker wird
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    
    # Normalisierte Distanz (0 im Zentrum, 1 an den Ecken)
    normalized_distance = distance / max_distance
    
    # Coma-Effekt ist am Rand stärker
    coma_mask = normalized_distance * strength
    
    # Finde helle Pixel (Sterne)
    threshold = np.percentile(image, 98)
    bright_points = np.where(image > threshold)
    
    for i in range(len(bright_points[0])):
        y, x = bright_points[0][i], bright_points[1][i]
        
        # Coma-Effekt verstärkt sich zum Rand hin
        local_coma = coma_mask[y, x]
        if local_coma > 0.2:  # Nur an Punkten mit ausreichendem Coma-Effekt
            brightness = image[y, x]
            
            # Ziehe den Stern in Richtung Rand (radialer Effekt)
            direction_x = x - center_x
            direction_y = y - center_y
            
            if direction_x != 0 or direction_y != 0:
                # Normalisiere Richtung
                length = np.sqrt(direction_x**2 + direction_y**2)
                direction_x /= length
                direction_y /= length
                
                # Erstelle einen elliptischen/asymmetrischen Fleck
                for r in range(1, int(10 * local_coma)):
                    for theta in range(0, 360, 10):
                        # Asymmetrische Form in Richtung des Randes
                        a = 1 + 2 * local_coma  # Längere Achse in Randrichtung
                        b = 1
                        
                        # Koordinaten
                        dx = r * a * np.cos(np.radians(theta))
                        dy = r * b * np.sin(np.radians(theta))
                        
                        # Drehe die Ellipse in Randrichtung
                        angle = np.arctan2(direction_y, direction_x)
                        rotated_x = dx * np.cos(angle) - dy * np.sin(angle)
                        rotated_y = dx * np.sin(angle) + dy * np.cos(angle)
                        
                        new_x = int(x + rotated_x)
                        new_y = int(y + rotated_y)
                        
                        if 0 <= new_x < width and 0 <= new_y < height:
                            # Intensität nimmt mit Abstand ab
                            intensity = brightness * (1 - (r / (10 * local_coma)))
                            result[new_y, new_x] = max(result[new_y, new_x], intensity)
    
    return result

def create_unround_stars(image, max_ellipticity=0.5):
    """Erstellt unrunde Sterne durch Verzerrung."""
    result = image.copy()
    height, width = image.shape[:2]
    
    # Finde helle Pixel (potentielle Sterne)
    threshold = np.percentile(image, 98)
    bright_points = np.where(image > threshold)
    
    # Zufällig einige helle Punkte auswählen
    if len(bright_points[0]) > 0:
        num_points = min(30, len(bright_points[0]))
        indices = np.random.choice(len(bright_points[0]), num_points, replace=False)
        
        for i in indices:
            y, x = bright_points[0][i], bright_points[1][i]
            brightness = image[y, x]
            
            # Zufällige Elliptizität und Rotation
            ellipticity = random.uniform(0.1, max_ellipticity)
            angle = random.uniform(0, 2 * np.pi)
            
            # Parameter für die elliptische Verzerrung
            a = random.randint(3, 8)  # Große Achse
            b = int(a * (1 - ellipticity))  # Kleine Achse
            
            # Erstelle eine elliptische Maske
            for dy in range(-a, a+1):
                for dx in range(-a, a+1):
                    # Formel für Ellipse
                    rotated_dx = dx * np.cos(angle) - dy * np.sin(angle)
                    rotated_dy = dx * np.sin(angle) + dy * np.cos(angle)
                    
                    if (rotated_dx / a)**2 + (rotated_dy / b)**2 <= 1:
                        new_y, new_x = y + dy, x + dx
                        if 0 <= new_y < height and 0 <= new_x < width:
                            # Gaussian-ähnliches Intensitätsprofil
                            distance = np.sqrt((rotated_dx / a)**2 + (rotated_dy / b)**2)
                            intensity = brightness * np.exp(-4 * distance**2)
                            result[new_y, new_x] = max(result[new_y, new_x], intensity)
    
    return result

def process_image_pair(starry_image, starless_image=None, mask=None, augment=True):
    """
    Verarbeitet ein Bild-Paar für das Training mit spezifischen Augmentierungen
    für Spike- und Coma-Effekte.
    """
    # Bildgröße anpassen
    starry_resized = cv2.resize(starry_image, (IMG_SIZE, IMG_SIZE))
    
    if starless_image is None and mask is None:
        # Für synthetische Daten ohne Maske: Erstelle sternenlose Version durch Unschärfe
        blurred = cv2.GaussianBlur(starry_resized, (21, 21), 0)
        median_filtered = cv2.medianBlur(blurred, 21)
        starless_resized = median_filtered
    elif starless_image is None and mask is not None:
        # Mit Maske: Erstelle sternenlose Version durch Inpainting
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        starless_resized = apply_mask_to_remove_stars(starry_resized, mask_resized)
    else:
        # Mit vorhandenem sternenlosen Bild
        starless_resized = cv2.resize(starless_image, (IMG_SIZE, IMG_SIZE))
    
    if augment and DATA_AUGMENTATION:
        # Zufällige Rotation und Spiegelung
        k_rot = random.randint(0, 3)  # 0, 90, 180, oder 270 Grad
        starry_resized = np.rot90(starry_resized, k_rot)
        starless_resized = np.rot90(starless_resized, k_rot)
        
        if random.random() > 0.5:
            starry_resized = np.fliplr(starry_resized)
            starless_resized = np.fliplr(starless_resized)
            
        if random.random() > 0.5:
            starry_resized = np.flipud(starry_resized)
            starless_resized = np.flipud(starless_resized)
        
        # Spezielle Augmentierungen für Sternabberationen
        if random.random() > 0.3:
            # Diffraktionsspikes hinzufügen
            starry_resized = simulate_spikes(starry_resized)
            
        if random.random() > 0.3:
            # Coma-Effekt hinzufügen
            starry_resized = simulate_coma(starry_resized)
            
        if random.random() > 0.3:
            # Unrunde Sterne erstellen
            starry_resized = create_unround_stars(starry_resized)
    
    # Erweitere die Dimensionen für TensorFlow (batch_size, height, width, channels)
    starry_resized = np.expand_dims(starry_resized, axis=-1)
    starless_resized = np.expand_dims(starless_resized, axis=-1)
    
    return starry_resized, starless_resized

def load_dataset(data_dir, is_training=True):
    """Lädt Trainingsdaten und erstellt einen tf.data.Dataset."""
    # Suche nach allen Bilddateien (FITS und PNG)
    starry_paths = []
    for ext in ['.fits', '.png', '.jpg', '.jpeg']:
        starry_paths.extend(glob.glob(os.path.join(data_dir, 'starry', f'*{ext}')))
    starry_paths = sorted(starry_paths)
    
    # Suche nach sternenfreien Bildern
    starless_paths = []
    for ext in ['.fits', '.png', '.jpg', '.jpeg']:
        starless_paths.extend(glob.glob(os.path.join(data_dir, 'starless', f'*{ext}')))
    starless_paths = sorted(starless_paths)
    
    # Suche nach Masken (wenn verfügbar)
    mask_paths = []
    if USE_MASKS:
        for ext in ['.fits', '.png', '.jpg', '.jpeg']:
            mask_paths.extend(glob.glob(os.path.join(data_dir, 'masks', f'*{ext}')))
        mask_paths = sorted(mask_paths)
    
    if not starry_paths:
        raise ValueError(f"Keine Starry-Bilder gefunden in {os.path.join(data_dir, 'starry')}")
    
    print(f"Gefundene Bilder: {len(starry_paths)} starry, {len(starless_paths)} starless, {len(mask_paths)} masks")
    
    image_pairs = []
    for idx, starry_path in enumerate(tqdm(starry_paths, desc="Lade Bilder")):
        # Lade starry image
        starry_img = load_image(starry_path)
        if starry_img is None:
            continue
        
        # Versuche, ein entsprechendes sternenfreies Bild zu finden
        starless_img = None
        if idx < len(starless_paths):
            # Versuche, anhand des Namens ein passendes Bild zu finden
            base_name = os.path.basename(starry_path).split('.')[0]
            matching_starless = [p for p in starless_paths if base_name in p]
            
            if matching_starless:
                starless_img = load_image(matching_starless[0])
        
        # Versuche, eine entsprechende Maske zu finden
        mask_img = None
        if USE_MASKS and mask_paths:
            # Versuche, anhand des Namens eine passende Maske zu finden
            base_name = os.path.basename(starry_path).split('.')[0]
            matching_masks = [p for p in mask_paths if base_name in p]
            
            if matching_masks:
                mask_img = load_image(matching_masks[0], normalize=True)
        
        # Verarbeite die Bilder
        starry_processed, starless_processed = process_image_pair(
            starry_img, starless_img, mask_img, augment=is_training
        )
        
        image_pairs.append((starry_processed, starless_processed))
    
    if not image_pairs:
        raise ValueError("Keine gültigen Bildpaare gefunden!")
    
    # Erstelle ein tf.data.Dataset
    def generator():
        for pair in image_pairs:
            yield pair
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32)
        )
    )
    
    if is_training:
        # Für das Training: Mischen, Batching und Präfetching
        dataset = dataset.shuffle(buffer_size=min(len(image_pairs), 1000))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # Für die Validierung: Nur Batching und Präfetching
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(image_pairs)

def attention_gate(x, g, filters):
    """Attention Gate für U-Net."""
    init = tf.keras.initializers.GlorotNormal()
    
    # Merkmalsprojektion für x und g
    theta_x = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer=init)(x)
    phi_g = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer=init)(g)
    
    # Fusion von x und g durch Addition
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    
    # Projektion auf skalaren Attention-Koeffizienten
    psi_f = layers.Conv2D(1, (1, 1), padding='same', kernel_initializer=init)(f)
    
    # Erzeugung der Attention-Map
    rate = layers.Activation('sigmoid')(psi_f)
    
    # Skalieren der Eingabe mit der Attention-Map
    att_x = layers.multiply([x, rate])
    
    return att_x

def build_unet_with_attention():
    """Verbesserte U-Net-Architektur mit Attention-Mechanismen."""
    # Input
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder mit Attention
    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(up6)
    att6 = attention_gate(conv4, up6, 512)
    merge6 = layers.Concatenate(axis=3)([att6, up6])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(up7)
    att7 = attention_gate(conv3, up7, 256)
    merge7 = layers.Concatenate(axis=3)([att7, up7])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(up8)
    att8 = attention_gate(conv2, up8, 128)
    merge8 = layers.Concatenate(axis=3)([att8, up8])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(up9)
    att9 = attention_gate(conv1, up9, 64)
    merge9 = layers.Concatenate(axis=3)([att9, up9])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class CombinedLoss(tf.keras.losses.Loss):
    """Kombinierte Loss-Funktion mit MSE und SSIM."""
    def __init__(self, alpha=0.8, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.mse = MeanSquaredError()
    
    def call(self, y_true, y_pred):
        # MSE-Anteil
        mse_loss = self.mse(y_true, y_pred)
        
        # SSIM-Anteil (1 - SSIM, da SSIM maximiert werden soll)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
        
        # Kombinierte Loss-Funktion
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

def create_perceptual_loss_model():
    """Erstellt ein Modell für Perceptual Loss basierend auf VGG16."""
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Ausgewählte Layer für Perceptual Loss
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    # Model erstellen, das die Features extrahiert
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    
    return model

class PerceptualLoss(tf.keras.losses.Loss):
    """Perceptual Loss basierend auf VGG16-Features."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.perceptual_model = create_perceptual_loss_model()
        self.mse = MeanSquaredError()
    
    def call(self, y_true, y_pred):
        # Konvertiere Graustufenbilder zu RGB (für VGG16)
        y_true_rgb = tf.concat([y_true, y_true, y_true], axis=-1)
        y_pred_rgb = tf.concat([y_pred, y_pred, y_pred], axis=-1)
        
        # Normalisiere Bilder für VGG16
        y_true_rgb = tf.keras.applications.vgg16.preprocess_input(y_true_rgb * 255.0)
        y_pred_rgb = tf.keras.applications.vgg16.preprocess_input(y_pred_rgb * 255.0)
        
        # Berechne Features
        true_features = self.perceptual_model(y_true_rgb)
        pred_features = self.perceptual_model(y_pred_rgb)
        
        # Berechne Loss über alle Feature-Maps
        loss = 0
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss += self.mse(true_feat, pred_feat)
        
        return loss

def train_model():
    """Hauptfunktion zum Training des Modells."""
    # Mixed Precision aktivieren für schnelleres Training
    if MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed Precision aktiviert")
    
    # Datasets laden
    train_dataset, num_train_samples = load_dataset(TRAINING_DATA_PATH, is_training=True)
    
    # Prüfen, ob Validierungsdaten existieren
    validation_starry_path = os.path.join(VALIDATION_DATA_PATH, 'starry')
    has_validation_data = os.path.exists(validation_starry_path) and len(os.listdir(validation_starry_path)) > 0
    
    if has_validation_data:
        print("Lade separate Validierungsdaten...")
        val_dataset, num_val_samples = load_dataset(VALIDATION_DATA_PATH, is_training=False)
    else:
        print("Keine Validierungsdaten gefunden. Verwende 20% der Trainingsdaten für Validierung...")
        # Teile das Trainings-Dataset in Training und Validierung auf
        train_size = int(0.8 * num_train_samples)
        val_size = num_train_samples - train_size
        
        train_dataset = train_dataset.shuffle(buffer_size=num_train_samples)
        val_dataset = train_dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.take(train_size)
        
        num_val_samples = val_size
    
    print(f"Trainingsdaten: {num_train_samples} Samples")
    print(f"Validierungsdaten: {num_val_samples} Samples")
    
    # Model erstellen
    model = build_unet_with_attention()
    print(model.summary())
    
    # Rest der Funktion bleibt unverändert...
    # Loss und Optimizer
    combined_loss = CombinedLoss(alpha=0.7)
    perceptual_loss = PerceptualLoss()
    
    def total_loss(y_true, y_pred):
        c_loss = combined_loss(y_true, y_pred)
        p_loss = perceptual_loss(y_true, y_pred)
        return c_loss + 0.1 * p_loss
    
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Kompilieren
    model.compile(
        optimizer=optimizer,
        loss=total_loss,
        metrics=['mse']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, 'model-{epoch:02d}-{val_loss:.4f}.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Model-Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Training starten
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # Finales Modell speichern
    model.save(os.path.join(MODEL_PATH, 'final_model.h5'))
    
    # Training-Historie plotten
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'training_history.png'))
    plt.close()
    
    return model, history


def predict_and_visualize(model, test_image_path, output_dir="results"):
    """Vorhersage und Visualisierung der Ergebnisse."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Lade Testbild
    starry_image = load_image(test_image_path)
    if starry_image is None:
        print(f"Konnte Testbild nicht laden: {test_image_path}")
        return
    
    # Verarbeite das Bild
    height, width = starry_image.shape
    starry_processed, _ = process_image_pair(starry_image, augment=False)
    
    # Vorhersage
    starless_predicted = model.predict(np.expand_dims(starry_processed, axis=0))[0]
    
    # Nachbearbeitung für Spikes und Artefakte
    starless_predicted = starless_predicted.squeeze()
    
    # Zurück zur ursprünglichen Größe skalieren
    starless_resized = cv2.resize(starless_predicted, (width, height))
    
    # Visualisieren
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(starry_image, cmap='viridis')
    plt.title('Original mit Sternen')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(starless_resized, cmap='viridis')
    plt.title('Vorhergesagt (ohne Sterne)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Differenzbild (entfernte Sterne)
    diff = np.clip(starry_image - starless_resized, 0, 1)
    plt.imshow(diff, cmap='hot')
    plt.title('Entfernte Sterne')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Speichern
    file_name = os.path.basename(test_image_path).split('.')[0]
    plt.savefig(os.path.join(output_dir, f"{file_name}_prediction.png"))
    plt.close()
    
    # FITS-Datei speichern
    if test_image_path.endswith('.fits'):
        fits.writeto(
            os.path.join(output_dir, f"{file_name}_starless.fits"),
            starless_resized,
            overwrite=True
        )
    else:
        # PNG oder JPEG speichern
        cv2.imwrite(
            os.path.join(output_dir, f"{file_name}_starless.png"),
            (starless_resized * 255).astype(np.uint8)
        )
    
    return starless_resized

if __name__ == "__main__":
    print("StarlessAI Training startet...")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU verfügbar: {tf.config.list_physical_devices('GPU')}")
    
    # GPU-Konfiguration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Training starten
    model, history = train_model()
    
    # Optional: Testbilder verarbeiten, wenn vorhanden
    if os.path.exists(TEST_DATA_PATH):
        test_images = []
        for ext in ['.fits', '.png', '.jpg', '.jpeg']:
            test_images.extend(glob.glob(os.path.join(TEST_DATA_PATH, f"*{ext}")))
        
        for img_path in test_images:
            predict_and_visualize(model, img_path)
