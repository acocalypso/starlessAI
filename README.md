# README

## FITS Image Processing and U-Net Model Training

### Overview
This project provides a Python-based pipeline to process FITS images, divide them into chunks, simulate background noise, and train a U-Net model for image processing tasks. The pipeline is designed to handle large datasets efficiently with GPU acceleration and memory optimization. It supports chunk-based dataset preparation, augmentation, and training with TensorFlow.

---

### Features

1. **FITS Image Loading**: Handles multi-dimensional FITS files and reduces them to 2D for processing.
2. **Chunking and Overlapping**: Divides large images into smaller, overlapping chunks for manageable input sizes.
3. **Background Simulation**: Simulates background data using Gaussian blur for training pairs.
4. **Dataset Preparation**: Efficiently processes FITS files into datasets using multiprocessing and augmentation.
5. **Memory Optimization**:
   - GPU memory growth configuration.
   - Mixed precision training for improved performance.
   - Garbage collection and memory cleanup callbacks.
6. **Custom Data Generators**: Handles large datasets by generating data in batches during training.
7. **U-Net Model**: Implements a U-Net architecture tailored for image processing tasks.

---

### Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy
- Astropy
- SciPy
- scikit-learn
- GPU with CUDA support (optional for faster training)

---

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/acocalypso/starlessAI
   cd starlessAI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

#### 1. Preparing the Dataset
Place your FITS files in the `input` folder and run the script to process them:
```bash
python train.py
```

The script will:
- Load and preprocess FITS files.
- Divide images into chunks.
- Augment data with flips and rotations.
- Save chunks to disk for training and validation.

#### 2. Training the Model
If no pre-trained model exists, the script will initialize a new U-Net model and start training using the prepared dataset. Training checkpoints are saved automatically.

#### 3. Resuming Training
If a checkpoint exists, the script will load the model and resume training.

#### 4. Model Evaluation
After training, the model can be evaluated on the validation dataset. Metrics like Mean Absolute Error (MAE) and loss are logged.

---

### Code Structure

- **Helper Functions**:
  - `load_fits_image`: Loads and preprocesses FITS images.
  - `divide_into_chunks`: Divides images into overlapping chunks.
  - `simulate_background`: Generates background data for training pairs.
- **Dataset Preparation**:
  - `prepare_dataset`: Processes images into training and validation datasets.
  - `save_chunks`: Saves chunks to disk for memory-efficient training.
- **Data Generators**:
  - `ChunkDataGenerator`: Provides chunk batches for training.
- **U-Net Model**:
  - Implements a custom U-Net architecture with an encoder-decoder structure.
- **Training**:
  - Uses TensorFlow's `fit` API with callbacks for early stopping, learning rate reduction, and memory cleanup.

---

### Model Architecture

The U-Net model is designed with:
- **Encoder**: Sequential convolutional and pooling layers.
- **Bottleneck**: A dense layer bridging the encoder and decoder.
- **Decoder**: Up-sampling and concatenation layers to reconstruct the input image.

---

### Configuration

Key parameters can be adjusted in the script:
- `chunk_size`: Size of image chunks (default: 128).
- `overlap`: Overlap between chunks (default: 32).
- `batch_size`: Batch size during training (default: 16).
- `epochs`: Number of training epochs (default: 50).
- `input_folder`: Path to FITS files (default: `input`).
- `chunk_dir`: Directory to save training chunks.
- `val_chunk_dir`: Directory to save validation chunks.

---

### Example

1. Place FITS files in the `input` folder.
2. Run the script:
   ```bash
   python train.py
   ```
3. Monitor training progress. The best model checkpoint is saved to `chunk_based_unet.keras`.

---

### Notes

- Ensure enough disk space for saving chunks and training logs.
- GPU acceleration is recommended for faster training.
- Mixed precision training improves performance on modern GPUs.

---

### Future Enhancements

- Add support for 3D FITS images.
- Integrate additional augmentation techniques.
- Implement a GUI for easier dataset preparation.

---
