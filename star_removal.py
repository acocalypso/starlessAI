import os
import argparse
import numpy as np
import tensorflow as tf
from astropy.io import fits
import cv2
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, PercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize

def parse_arguments():
    parser = argparse.ArgumentParser(description='StarlessAI: Entfernt Sterne aus astronomischen Bildern')
    parser.add_argument('--input', type=str, required=True, help='Eingabeordner mit Bildern')
    parser.add_argument('--output', type=str, required=True, help='Ausgabeordner für Ergebnisse')
    parser.add_argument('--model', type=str, default='models/starless_model/final_model.keras', 
                        help='Pfad zum trainierten Modell')
    parser.add_argument('--threshold', type=float, default=0.2, 
                        help='Schwellwert für Sternmaske (0.0-1.0)')
    parser.add_argument('--inpaint_radius', type=int, default=3, 
                        help='Radius für Inpainting-Algorithmus')
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='Größe der Bildabschnitte für die Verarbeitung')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Überlappung zwischen Bildabschnitten')
    return parser.parse_args()

def is_linear(image_data, image_path):
    """Verbesserte Erkennung, ob ein Bild linear ist."""
    # FITS sind typischerweise linear
    if image_path.lower().endswith(('.fits', '.fit')):
        return True
    
    # PNG-Dateien sind in der Regel bereits gestretcht
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return False
    
    # Für TIFF-Dateien: Prüfe Histogramm
    if image_data.dtype == np.uint8:
        # 8-bit Bilder sind in der Regel nicht linear
        return False
    
    # Prüfe Histogramm-Verteilung nur für andere Formate
    hist, bins = np.histogram(image_data.flatten(), bins=256)
    if np.sum(hist[:128]) > 0.9 * np.sum(hist):
        # Wenn 90% der Pixel im unteren Helligkeitsbereich liegen, 
        # ist das Bild wahrscheinlich linear
        return True
    
    return False


def load_image(file_path):
    """Lädt ein Bild im unterstützten Format mit Farberhaltung."""
    try:
        if file_path.lower().endswith(('.fits', '.fit')):
            # FITS-Datei laden
            with fits.open(file_path) as hdul:
                image = hdul[0].data
                header = hdul[0].header
                
                # Wenn Bild 3-dimensional ist, nehme das erste Bild
                if len(image.shape) > 2:
                    image = image[0]
                    
                # Negative Werte auf 0 setzen
                image = np.maximum(image, 0)
                
                # Konvertiere zu float32
                image = image.astype(np.float32)
                
                return image, header, True  # Linear
        else:
            # PNG, TIFF oder andere Bildformate - Farbe erhalten
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            # Alpha-Kanal entfernen, falls vorhanden
            if image is not None and len(image.shape) > 2 and image.shape[2] == 4:
                image = image[:,:,:3]
            
            # Konvertiere zu float32, aber behalte Farbkanäle bei
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
                linear = True
            else:
                image = image.astype(np.float32) / 255.0
                linear = is_linear(image, file_path)
            
            return image, None, linear
            
    except Exception as e:
        print(f"Fehler beim Laden von {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def prestretch_image(image):
    """Wendet einen Stretch auf lineare Daten an."""
    # Logarithmischer Stretch
    stretched = np.log1p(image * 100) / np.log(101)
    
    # Normalisieren auf [0,1]
    min_val = np.min(stretched)
    max_val = np.max(stretched)
    if max_val > min_val:
        stretched = (stretched - min_val) / (max_val - min_val)
    
    return stretched

def unstretch_image(stretched_image, original_image):
    """Konvertiert ein gestretchtes Bild zurück in den linearen Bereich."""
    # Skalieren auf den Bereich des Originalbildes
    min_orig = np.min(original_image)
    max_orig = np.max(original_image)
    
    # Inverse des logarithmischen Stretches
    unstretch_factor = 100
    unstretch = (np.exp(stretched_image * np.log(unstretch_factor + 1)) - 1) / unstretch_factor
    
    # Skalieren auf den Originalbereich
    unstretch = unstretch * (max_orig - min_orig) + min_orig
    
    return unstretch

def extract_chunks(image, chunk_size=256, overlap=32):
    """Teilt ein Bild in überlappende Chunks auf."""
    # Prüfe, ob das Bild farbig ist
    if len(image.shape) > 2:
        height, width, channels = image.shape
        is_color = True
    else:
        height, width = image.shape
        is_color = False
        
    chunks = []
    positions = []
    
    # Berechne effektive Schrittweite
    stride = chunk_size - overlap
    
    # Iteriere über das Bild in Schritten von stride
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Stelle sicher, dass der Chunk nicht über den Bildrand hinausgeht
            end_y = min(y + chunk_size, height)
            end_x = min(x + chunk_size, width)
            
            # Stelle sicher, dass der Chunk die richtige Größe hat
            start_y = max(0, end_y - chunk_size)
            start_x = max(0, end_x - chunk_size)
            
            # Extrahiere den Chunk
            chunk = image[start_y:end_y, start_x:end_x]
            
            # Wenn der Chunk kleiner als chunk_size ist, fülle ihn auf
            if chunk.shape[0] < chunk_size or chunk.shape[1] < chunk_size:
                padded_chunk = np.zeros((chunk_size, chunk_size), dtype=chunk.dtype)
                padded_chunk[:chunk.shape[0], :chunk.shape[1]] = chunk
                chunk = padded_chunk
            
            chunks.append(chunk)
            positions.append((start_y, start_x, end_y, end_x))
    
    return chunks, positions

def process_chunk(chunk, model):
    """Verarbeitet einen einzelnen Chunk mit dem Modell."""
    # Erweitern für Batch-Dimension und Kanal
    chunk_batch = np.expand_dims(np.expand_dims(chunk, axis=0), axis=-1)
    
    # Modellvorhersage
    starless_pred = model.predict(chunk_batch, verbose=0)[0]
    
    # Zurück zu 2D
    starless_pred = starless_pred.squeeze()
    
    return starless_pred

def blend_chunks(chunks, positions, original_shape, weights=None):
    """Fügt Chunks mit gewichteter Überlappung wieder zusammen."""
    height, width = original_shape
    result = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # Verbesserte Gewichtungsfunktion mit weicherem Übergang
    if weights is None:
        chunk_size = chunks[0].shape[0]
        y, x = np.ogrid[:chunk_size, :chunk_size]
        center_y, center_x = chunk_size // 2, chunk_size // 2
        
        # Verwende eine Gaußsche Gewichtung für weichere Übergänge
        sigma = chunk_size / 6  # Parameter für die Breite der Gaußkurve
        weights = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Normalisiere die Gewichte
        weights = weights / np.max(weights)
    
    # Füge die Chunks zusammen
    for chunk, (start_y, start_x, end_y, end_x) in zip(chunks, positions):
        # Extrahiere den tatsächlichen Bereich aus dem Chunk
        actual_height = end_y - start_y
        actual_width = end_x - start_x
        actual_chunk = chunk[:actual_height, :actual_width]
        actual_weights = weights[:actual_height, :actual_width]
        
        # Addiere gewichteten Chunk zum Ergebnis
        result[start_y:end_y, start_x:end_x] += actual_chunk * actual_weights
        weight_map[start_y:end_y, start_x:end_x] += actual_weights
    
    # Normalisiere durch die Gewichtssumme
    # Vermeide Division durch Null
    weight_map = np.maximum(weight_map, 1e-10)
    result = result / weight_map
    
    return result


def process_image_with_chunks(image, model, chunk_size=256, overlap=64, threshold=0.2, inpaint_radius=3):
    """Verarbeitet ein Bild in Chunks und erzeugt Sternmaske und sternenfreies Bild mit Farberhaltung."""
    is_color = len(image.shape) > 2 and image.shape[2] >= 3
    
    if is_color:
        # Verarbeite jeden Farbkanal separat
        starless_channels = []
        binary_masks = []
        
        for channel in range(3):
            channel_data = image[:,:,channel]
            chunks, positions = extract_chunks(channel_data, chunk_size, overlap)
            
            # Verarbeite jeden Chunk
            processed_chunks = []
            for chunk in tqdm(chunks, desc=f"Verarbeite Kanal {channel}", leave=False):
                processed_chunk = process_chunk(chunk, model)
                processed_chunks.append(processed_chunk)
            
            # Füge die Chunks zusammen
            starless_channel = blend_chunks(processed_chunks, positions, channel_data.shape)
            
            # Erzeuge Sternmaske
            stars_mask = np.clip(channel_data - starless_channel, 0, 1)
            binary_mask = (stars_mask > threshold).astype(np.uint8)
            
            starless_channels.append(starless_channel)
            binary_masks.append(binary_mask)
        
        # Kombiniere die Masken
        final_mask = np.maximum.reduce(binary_masks)
        
        # Verbessere die Maske
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # Inpainting auf dem Originalbild
        img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        mask_8bit = (final_mask * 255).astype(np.uint8)
        inpainted = cv2.inpaint(img_8bit, mask_8bit, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        inpainted = inpainted.astype(np.float32) / 255.0
        
        # Kombiniere die Ergebnisse
        starless_image = np.stack(starless_channels, axis=-1)
        final_starless = np.where(np.expand_dims(final_mask, -1) > 0, inpainted, starless_image)
        
        return final_starless, final_mask
    else:
        # Graustufenbild direkt verarbeiten
        chunks, positions = extract_chunks(image, chunk_size, overlap)
    
        # Verarbeite jeden Chunk
        processed_chunks = []
        for chunk in tqdm(chunks, desc="Verarbeite Chunks", leave=False):
            processed_chunk = process_chunk(chunk, model)
            processed_chunks.append(processed_chunk)
        
        # Verbesserte Gewichtung für das Zusammenfügen
        chunk_size = chunks[0].shape[0]
        y, x = np.ogrid[:chunk_size, :chunk_size]
        center_y, center_x = chunk_size // 2, chunk_size // 2
        
        # Verwende eine Gaußsche Gewichtung für weichere Übergänge
        sigma = chunk_size / 6  # Parameter für die Breite der Gaußkurve
        weights = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Normalisiere die Gewichte
        weights = weights / np.max(weights)
        
        # Für Graustufenbilder
        starless_image = blend_chunks(processed_chunks, positions, image.shape, weights)
        
        # Sternmaske erzeugen
        stars_mask = np.clip(image - starless_image, 0, 1)
        binary_mask = (stars_mask > threshold).astype(np.uint8)
        
        # Maske verbessern
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Inpainting
        img_8bit = np.clip(image * 255, 0, 255).astype(np.uint8)
        mask_8bit = binary_mask * 255
        inpainted = cv2.inpaint(img_8bit, mask_8bit, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        inpainted = inpainted.astype(np.float32) / 255.0
        
        # Kombiniere Ergebnisse
        final_starless = np.where(binary_mask > 0, inpainted, starless_image)
        
        return final_starless, binary_mask


def save_result(image, file_path, original_header=None):
    """Speichert ein Bild im entsprechenden Format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_path.lower().endswith(('.fits', '.fit')):
        # Als FITS speichern
        if original_header:
            fits.writeto(file_path, image, header=original_header, overwrite=True)
        else:
            fits.writeto(file_path, image, overwrite=True)
    else:
        # Als PNG oder TIFF speichern
        if image.dtype == np.float32:
            if np.max(image) <= 1.0:
                # [0,1] Bereich
                image_8bit = (image * 255).astype(np.uint8)
            else:
                # Höherer Bereich (16-bit)
                image_16bit = np.clip(image, 0, 65535).astype(np.uint16)
                if file_path.lower().endswith('.png'):
                    # PNG unterstützt 16-bit
                    cv2.imwrite(file_path, image_16bit)
                    return
                else:
                    # Für andere Formate auf 8-bit reduzieren
                    image_8bit = (image_16bit / 256).astype(np.uint8)
        else:
            image_8bit = image
            
        cv2.imwrite(file_path, image_8bit)

def visualize_results(original, starless, mask, output_path):
    """Erzeugt eine Visualisierung der Ergebnisse."""
    plt.figure(figsize=(15, 5))
    
    # Für Farbbilder
    if len(original.shape) > 2:
        plt.subplot(1, 3, 1)
        plt.imshow(np.clip(original, 0, 1))
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(starless, 0, 1))
        plt.title('Ohne Sterne')
        plt.axis('off')
    else:
        # Für Graustufenbilder
        norm = ImageNormalize(stretch=SqrtStretch(), interval=PercentileInterval(98.0))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap='viridis', norm=norm)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(starless, cmap='viridis', norm=norm)
        plt.title('Ohne Sterne')
        plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Sternmaske')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_arguments()
    
    # Modell laden
    print(f"Lade Modell von {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)
    
    # Eingabeordner überprüfen
    input_dir = args.input
    if not os.path.exists(input_dir):
        print(f"Eingabeordner {input_dir} existiert nicht!")
        return
    
    # Ausgabeordner erstellen
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Unterstützte Dateiformate
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit']
    
    # Alle Bilder im Eingabeordner finden
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_files:
        print(f"Keine unterstützten Bilddateien im Ordner {input_dir} gefunden!")
        return
    
    print(f"Gefunden: {len(image_files)} Bilder zum Verarbeiten")
    
    # Bilder verarbeiten
    for image_path in tqdm(image_files, desc="Verarbeite Bilder"):
        # Dateiname ohne Pfad und Erweiterung
        base_name = os.path.basename(image_path).rsplit('.', 1)[0]
        
        # Bild laden
        image, header, is_linear = load_image(image_path)
        if image is None:
            print(f"Überspringe {image_path}: Konnte nicht geladen werden")
            continue
        
        # Original speichern für später
        original_image = image.copy()
        
        # Wenn linear, Prestretch anwenden
        if is_linear:
            print(f"{image_path} ist linear, wende Prestretch an")
            image = prestretch_image(image)
        
        # Sterne entfernen und Maske erstellen mit Chunk-basierter Verarbeitung
        starless, mask = process_image_with_chunks(
            image, model, 
            chunk_size=args.chunk_size, 
            overlap=args.overlap,
            threshold=args.threshold, 
            inpaint_radius=args.inpaint_radius
        )
        
        # Wenn linear, zurück in den linearen Bereich konvertieren
        if is_linear:
            print(f"Konvertiere {image_path} zurück in den linearen Bereich")
            starless = unstretch_image(starless, original_image)
        
        # Ausgabepfade
        ext = os.path.splitext(image_path)[1]
        starless_path = os.path.join(output_dir, f"{base_name}_starless{ext}")
        mask_path = os.path.join(output_dir, f"{base_name}_stars_mask{ext}")
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        
        # Ergebnisse speichern
        save_result(starless, starless_path, header)
        save_result(mask.astype(np.float32), mask_path)
                
        # Visualisierung erstellen
        visualize_results(original_image, starless, mask, vis_path)
                
        print(f"Verarbeitet: {image_path} -> {starless_path}, {mask_path}")
            
    print(f"Alle Bilder verarbeitet. Ergebnisse in {output_dir}")
        
if __name__ == "__main__":
    main()
