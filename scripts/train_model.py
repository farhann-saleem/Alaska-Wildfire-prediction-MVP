# 
# SCRIPT: train_model.py
# PURPOSE: Loads patched data in-memory ||Trains using TensorFlow || Evaluates performance on test-set.
#    Simple CNN baseline model(failed)
#    Enhanced CNN baseline model (currently active )

# DIAGNOSIS: Used to confirm failure to generalize due to class imbalance (Recall=0.00).

# --- Imports ---
import os
import pandas as pd
import numpy as np
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score

# --- Configuration ---
PROJECT_ROOT = r"G:\UAAS\Wildlife-prediction-try02"  
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
METADATA_CSV = os.path.join(PROJECT_ROOT, "data", "patch_metadata.csv")


# Constants
SENTINEL_TIF = os.path.join(DATA_DIR, "s2_2021_06_input_10m.tif")
PATCH_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


# --- 1. Data Loading and Patch Extraction 

def load_all_patches_into_memory(metadata_path, sentinel_path, patch_size=64):
    """Loads the entire image and metadata, then cuts the patches from the memory array."""
    print("Loading large GeoTIFF into memory and extracting patches...")
    
    # 1.1 Load Data [came from preprocess.py]
    df = pd.read_csv(metadata_path)
    
    try:
        with rasterio.open(sentinel_path) as src:
            # Load the entire GeoTIFF as a NumPy array (bands, H, W)
            full_image_array = src.read()
            img_width, img_height = src.width, src.height
    except rasterio.RasterioIOError:
        print(f"FATAL ERROR: Could not open {sentinel_path}. Check file path.")
        return None, None, None
        
    print(f"Full image array loaded with shape (C, H, W): {full_image_array.shape}")
    
    # NumPy array format from (C, H, W) to (H, W, C) for TensorFlow

    full_image_array = np.transpose(full_image_array, (1, 2, 0)) # Now shape is (H, W, C)

    # 2. Extract Patches from the in-memory array
    patch_list = []
    
    # Recalculating  patch start indices based on the sequential patch_id from preprocess.py
    num_patches_x = img_width // patch_size
    
    # Iterate same way as preprocess.py to avoid index mismatch
    patch_list = []
    label_list = []
    patch_count = 0
    nan_count = 0
    
    for y in range(0, img_height, patch_size):
        for x in range(0, img_width, patch_size):
            if x + patch_size > img_width or y + patch_size > img_height:
                continue 
            
            # Cut the patch from the array (H, W, C) format
            patch = full_image_array[y:y + patch_size, 
                                     x:x + patch_size, 
                                     :]
            
            # Normalize the pixel values (Sentinel-2 SR is scaled to 10k)
            patch = patch / 10000.0
            
            # SKIP patches with NaN values
            if np.isnan(patch).any():
                nan_count += 1
                patch_count += 1
                continue
            
            patch_list.append(patch)
            # Get label from metadata (patches extracted in same order as preprocess.py)
            if patch_count < len(df):
                label_list.append(df.iloc[patch_count]['burn_label'])
            patch_count += 1
    
    print("\n=== PATCH EXTRACTION ===")
    print(f"Total patches: {patch_count}")
    print(f"Valid patches: {len(patch_list)}")
    print(f"NaN patches skipped: {nan_count}")
    print(f"Labels loaded: {len(label_list)}")
    
    if len(patch_list) != len(label_list):
        print(f"ERROR: Patch/Label mismatch! {len(patch_list)} patches vs {len(label_list)} labels")
        return None, None, None
    
    print(f"Sample patch 0 stats - Min: {patch_list[0].min():.6f}, Max: {patch_list[0].max():.6f}, Mean: {patch_list[0].mean():.6f}")
    if len(patch_list) > 100:
        print(f"Sample patch 100 stats - Min: {patch_list[100].min():.6f}, Max: {patch_list[100].max():.6f}, Mean: {patch_list[100].mean():.6f}")
    
    # Check class distribution
    label_array = np.array(label_list)
    print(f"Class 0 (No-Burn): {np.sum(label_array == 0)}")
    print(f"Class 1 (Burn): {np.sum(label_array == 1)}")

    # Convert list of NumPy arrays to a single NumPy array (N, H, W, C)
    X = np.stack(patch_list).astype(np.float32)
    
    # Convert labels to NumPy array
    y = np.array(label_list)
    
    return X, y, df



#  SUMMARY : MODEL FAILED --Imbalanced dataset
# # --- 2. Baseline CNN Model Definition (TensorFlow/Keras) ---
# def create_simple_cnn(input_shape, num_classes=2):
#     """Creates a Sequential Keras model equivalent to the SimpleCNN baseline."""    
#     model = tf.keras.models.Sequential([
#         # Input Layer: Accepts patches of size (PATCH_SIZE, PATCH_SIZE, 3)
#         tf.keras.layers.InputLayer(input_shape=input_shape),
#         # Block 1: 
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         # Block 2:
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         # Classification Head:
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(num_classes, activation='softmax') # Softmax for 2 classes
#     ])
#     return model


# --- 2. Enhanced CNN Model Definition (TensorFlow/Keras) ---

def create_enhanced_cnn(input_shape, num_classes=2):
    """Creates an enhanced, deeper Keras model using residual blocks for better feature learning."""
    
    # residual block function for deeper learning
    def res_block(x, filters, kernel_size=3):
        y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
        x = tf.keras.layers.Conv2D(filters, 1)(x) 
        z = tf.keras.layers.Add()([x, y])
        return tf.keras.layers.ReLU()(z)

    # Input Layer: Accepts patches of size (PATCH_SIZE, PATCH_SIZE, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Initial Feature Extraction
    x = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x) # 32x32

    # Residual Block 1
    x = res_block(x, 32)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x) # 16x16

    # Residual Block 2 (Depper Features)
    x = res_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x) # 8x8
    
    # Classification Head:
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x) # Add dropout to prevent overfitting
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # Softmax for 2 classes

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# --- 3. Training and Evaluation ---

def train_and_evaluate_tf():
    # --- 1. Load and Split Data ---
    X_data, y_data, df = load_all_patches_into_memory(METADATA_CSV, SENTINEL_TIF, PATCH_SIZE)
    
    if X_data is None:
        return 
        
    # Split data: 75% train, 25% test. Stratify ensures burn patches are split evenly.
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42, stratify=y_data
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # --- 2. Initialize Model and Training components ---
    # Input shape is (H, W, C) -> (64, 64, 3)
    model = create_enhanced_cnn(input_shape=(PATCH_SIZE, PATCH_SIZE, 3))
    

    # Calculate class weights for imbalance (critical step for 1.7% positive class) to prevents the model from ignoring the rare 'Burn' class.
    class_counts = df['burn_label'].value_counts()
    total_samples = len(df)
    
    # Weights are calculated as: total_samples / (num_classes * class_count)
    weight_0 = total_samples / (2 * class_counts[0]) # Weight for No-Burn (0)
    weight_1 = total_samples / (2 * class_counts[1]) # Weight for Burn (1)
    
    # Scale down burn weight to prevent model collapse
    weight_1 = weight_1 * 0.3
    
    class_weights = {0: weight_0, 1: weight_1}
    
    print(f"Class Weights (0: No-Burn, 1: Burn): {class_weights}")
    
    # Compiling model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy', # Appropriate for integer labels (0, 1)
        metrics=['accuracy']
    )
    
    # --- 3. Training Loop ---
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=1 
    )
    
    print("Training finished.")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Loss progression: {[f'{l:.4f}' for l in history.history['loss']]}")

      # --- 4. Evaluation on Test Set ---
    y_pred_proba = model.predict(X_test)
     
     # DEBUG: Check actual probabilities
    print("\n=== DEBUG: Probability Distribution ===")
    print(f"Min prob for class 1: {y_pred_proba[:, 1].min():.6f}")
    print(f"Max prob for class 1: {y_pred_proba[:, 1].max():.6f}")
    print(f"Mean prob for class 1: {y_pred_proba[:, 1].mean():.6f}")
    print(f"Median prob for class 1: {np.median(y_pred_proba[:, 1]):.6f}")
     
     # Show distribution
    import matplotlib.pyplot as plt
    plt.hist(y_pred_proba[:, 1], bins=50)
    plt.xlabel('Probability of Burn Class')
    plt.savefig('prob_distribution.png')
    print("Saved probability distribution to prob_distribution.png")

    y_pred = np.argmax(y_pred_proba, axis=1)

    # --- 5. Calculate Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    
    print("\n--- Baseline Model Evaluation ---")
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Burned Class): {f1:.4f}") 
    print(f"Recall (Burned Class): {recall:.4f}")
    print("---------------------------------")
    
    return accuracy, f1, recall

if __name__ == '__main__':
    # Ensure TensorFlow only uses the necessary GPU memory
    # You can comment out this section if you don't have a GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Enable memory growth to avoid allocating all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU setup complete.")
        except RuntimeError as e:
            print(e)
            
    train_and_evaluate_tf()