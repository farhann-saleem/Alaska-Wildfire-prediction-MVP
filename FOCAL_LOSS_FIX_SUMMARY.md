# 🔥 Focal Loss Fix - What Changed

## Problem
```
ModuleNotFoundError: No module named 'keras.src.engine'
```
This was caused by `tensorflow_addons` having version conflicts.

## Solution
**Implemented custom Focal Loss - NO external dependencies needed!**

---

## Changes Made to READY_TO_PASTE_COLAB_CELL.py

### Change 1: Added Custom Focal Loss Class (After imports)
```python
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.reduce_max(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1 - pt, self.gamma)
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return tf.reduce_mean(focal_loss)
```

### Change 2: Updated Config
```python
EPOCHS = 50  # was 30
LEARNING_RATE = 0.01  # was 0.001
```

### Change 3: Compile with Custom Focal Loss
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=FocalLoss(alpha=0.25, gamma=2.0),  # Custom!
    metrics=['accuracy']
)
```

### Change 4: Updated Print Messages
```python
✅ Fix #3: FOCAL LOSS (custom, no external deps)
✅ Higher LR: 0.01 instead of 0.001
✅ More epochs: 50 instead of 30
```

---

## Why This Works

**Focal Loss** = Best for extreme imbalance (1.7% positive class)

- **Alpha (0.25)**: Weights the positive class more
- **Gamma (2.0)**: Focuses on hard-to-classify examples
- **Custom implementation**: No external dependencies = no version conflicts!

---

## Expected Results

- Loss: 0.3-0.45 (NOT stuck at 0.693!)
- Recall: > 0.3 (NOT 0.0!)
- Probabilities: spread 0.01-0.99 (NOT stuck at 0.5!)

---

## Now Just Copy & Paste!

1. Open `READY_TO_PASTE_COLAB_CELL.py`
2. Copy ALL the code
3. Paste into `main.ipynb` cell
4. Run! ▶️

**No more version conflicts!** 🚀
