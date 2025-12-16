# ============================================================
# CORRECTED TRAINING CELL FOR YOUR COLAB NOTEBOOK (TPU COMPATIBLE)
# ============================================================
# Copy this code into a cell in your main.ipynb

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score

# --- Constants ---
PATCH_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

def create_enhanced_cnn(input_shape, num_classes=2):
    """Enhanced CNN with residual blocks for imbalanced data."""
    
    def res_block(x, filters, kernel_size=3):
        y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
        x = tf.keras.layers.Conv2D(filters, 1)(x) 
        z = tf.keras.layers.Add()([x, y])
        return tf.keras.layers.ReLU()(z)

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = res_block(x, 32)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = res_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)  # Increased from 0.3
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Added extra dropout
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def train_with_fixes(X_data, y_data, df):
    """
    CORRECTED TRAINING FUNCTION WITH SOFTMAX COLLAPSE FIXES
    """
    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42, stratify=y_data
    )
    
    print(f"✓ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"✓ Train - Burn: {np.sum(y_train == 1)}, No-Burn: {np.sum(y_train == 0)}")
    
    # --- FIX #1: CALCULATE CORRECT CLASS WEIGHTS (NO SCALING) ---
    class_counts = df['burn_label'].value_counts()
    total_samples = len(df)
    
    weight_0 = total_samples / (2 * class_counts[0])
    weight_1 = total_samples / (2 * class_counts[1])
    
    # *** KEY FIX: DO NOT SCALE DOWN weight_1! ***
    class_weights = {0: weight_0, 1: weight_1}
    
    print(f"\n✓ Class Weights:")
    print(f"  - No-Burn (0): {weight_0:.4f}")
    print(f"  - Burn (1):    {weight_1:.4f}")
    print(f"  - Ratio (Burn/No-Burn): {weight_1/weight_0:.2f}x")
    
    # --- FIX #2: CONVERT TO ONE-HOT FOR BETTER LOSS HANDLING ---
    y_train_encoded = tf.keras.utils.to_categorical(y_train, 2)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, 2)
    
    # --- Initialize Model ---
    model = create_enhanced_cnn(input_shape=(PATCH_SIZE, PATCH_SIZE, 3))
    
    # --- FIX #3: USE CATEGORICAL_CROSSENTROPY WITH BALANCED WEIGHTS ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',  # Better for imbalanced + weighted
        metrics=['accuracy']
    )
    
    print(f"\n✓ Model compiled with categorical_crossentropy + class weights")
    
    # --- FIX #4: ADD CALLBACKS FOR CONVERGENCE ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # --- TRAINING WITH ALL FIXES ---
    print("\n✓ Starting training with fixes...\n")
    history = model.fit(
        X_train, y_train_encoded,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,  # Critical for imbalance
        validation_split=0.2,         # Monitor generalization
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training completed!")
    print(f"✓ Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"✓ Loss progression: {[f'{l:.4f}' for l in history.history['loss'][-5:]]}")
    
    # --- EVALUATION ---
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print("\n=== PROBABILITY ANALYSIS ===")
    print(f"Min prob for class 1: {y_pred_proba[:, 1].min():.6f}")
    print(f"Max prob for class 1: {y_pred_proba[:, 1].max():.6f}")
    print(f"Mean prob for class 1: {y_pred_proba[:, 1].mean():.6f}")
    print(f"Median prob for class 1: {np.median(y_pred_proba[:, 1]):.6f}")
    
    # Plot distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.hist(y_pred_proba[:, 1], bins=50, alpha=0.7)
    plt.xlabel('Probability of Burn Class')
    plt.ylabel('Count')
    plt.title('Probability Distribution (Should be SPREAD, not stuck at 0.5)')
    plt.savefig('prob_distribution_fixed.png', dpi=100)
    plt.close()
    print("✓ Saved probability distribution to prob_distribution_fixed.png")
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    print("\n=== CORRECTED MODEL EVALUATION ===")
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"F1 Score (Burn):       {f1:.4f}")
    print(f"Recall (Burn):         {recall:.4f}")
    print("=================================")
    
    return model, history, accuracy, f1, recall

# ============================================================
# USAGE IN YOUR NOTEBOOK:
# ============================================================
# 1. Load your patches as X_data, y_data (from preprocess)
# 2. Run:
#    model, history, acc, f1, recall = train_with_fixes(X_data, y_data, df)
# ============================================================
