import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, LayerNormalization, Dense, Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
import random
from scipy.ndimage import uniform_filter

# Check for GPU availability and configure GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

# Dice Coefficient and IoU Score
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# Optimized Wallis filter function
def optimized_wallis_filter(image, window_size, lambda_contrast=0.9):
    """Optimized Wallis sharpening algorithm with proper normalization."""
    image = image.astype(np.float32)
    
    # Global mean and standard deviation
    global_mean = np.mean(image)
    global_std = np.std(image)

    # Compute local mean and local variance using uniform_filter
    local_mean = uniform_filter(image, size=window_size)
    local_sqr_mean = uniform_filter(image**2, size=window_size)
    local_var = local_sqr_mean - local_mean**2
    local_std = np.sqrt(local_var)

    # Prevent division by very small numbers
    local_std = np.clip(local_std, 1e-5, None)

    # Wallis filter formula
    output_image = lambda_contrast * (image - local_mean) / local_std + (1 - lambda_contrast) * global_mean

    # Normalize output image to range [0, 255]
    min_val, max_val = np.min(output_image), np.max(output_image)
    output_image = 255 * (output_image - min_val) / (max_val - min_val)

    return output_image.astype(np.uint8)

# Preprocessing function with Wallis augmentation and blank mask removal
def preprocess_data(image_folder, mask_folder, augment=True):
    image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder)])
    mask_files = sorted([os.path.join(mask_folder, file) for file in os.listdir(mask_folder)])
    image_dict = {os.path.splitext(os.path.basename(file))[0]: file for file in image_files}
    mask_dict = {os.path.splitext(os.path.basename(file))[0]: file for file in mask_files}
    common_keys = set(image_dict.keys()).intersection(mask_dict.keys())
    
    images = [cv2.imread(image_dict[key], cv2.IMREAD_GRAYSCALE) for key in common_keys]
    masks = [cv2.imread(mask_dict[key], cv2.IMREAD_GRAYSCALE) for key in common_keys]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images = [clahe.apply(img) for img in images]
    
    augmented_images = []
    augmented_masks = []
    
    for img, mask in zip(images, masks):
        # Resizing
        img_resized = cv2.resize(img, (256, 256))
        mask_resized = cv2.resize(mask, (256, 256))
        
        # Remove images with blank masks (i.e., all pixel values are 0)
        if np.sum(mask_resized) == 0:
            continue
        
        # Keep the original image
        augmented_images.append(img_resized)
        augmented_masks.append(mask_resized)
        
        # Apply Wallis filter augmentation if augment=True
        if augment:
            # Select one random kernel size between 70 and 120
            kernel_size = random.randint(70, 120)
            img_augmented = optimized_wallis_filter(img_resized, window_size=kernel_size)
            
            # Add the augmented image and corresponding mask
            augmented_images.append(img_augmented)
            augmented_masks.append(mask_resized)
    
    # Convert lists to numpy arrays and normalize
    images = np.array(augmented_images).reshape(-1, 256, 256, 1)
    masks = np.array(augmented_masks).reshape(-1, 256, 256, 1)
    images = images / 255.0
    masks = masks / 255.0

    # Print the size of the dataset after preprocessing
    print(f"Total images: {images.shape[0]}, Total masks: {masks.shape[0]}")
    
    return images, masks

# Example usage
train_images, train_masks = preprocess_data(
    '/segmentation_mebo_om/new_data1/train/images',
    '/segmentation_mebo_om/new_data1/train/mask',
    augment=True
)
val_images, val_masks = preprocess_data(
    '/segmentation_mebo_om/new_data1/val/images',
    '/segmentation_mebo_om/new_data1/val/mask',
    augment=True
)

# Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.feed_forward = tf.keras.Sequential([
            Dense(dim * 4, activation='relu'),
            Dense(dim)
        ])
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

    def call(self, x):
        # Self-attention
        attn_output = self.attention(x, x)
        x = self.layer_norm1(x + attn_output)
        # Feed-forward
        ff_output = self.feed_forward(x)
        return self.layer_norm2(x + ff_output)

# TransUNet Model
def transunet_model(input_size=(256, 256, 1), num_heads=8, transformer_dim=256):
    inputs = Input(input_size)

    # Encoder Path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c5)

    # Transformer Bridge
    b5_shape = c5.shape  # Use KerasTensor.shape
    b5_flatten = Flatten()(c5)
    b5_reshape = Reshape((b5_shape[1] * b5_shape[2], b5_shape[3]))(b5_flatten)
    transformer = TransformerBlock(dim=transformer_dim, num_heads=num_heads)(b5_reshape)
    transformer_reshape = Reshape((b5_shape[1], b5_shape[2], transformer_dim))(transformer)

    # Decoder Path
    u6 = UpSampling2D((2, 2))(transformer_reshape)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(c9)

    # Output Layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=combined_loss, metrics=[dice_coefficient, iou_score])
    return model

# Create and compile TransUNet model
model = transunet_model()

# Define callbacks
checkpoint = ModelCheckpoint('logs_trans/transunet_best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir='logs_trans', histogram_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1, min_lr=1e-7)

# Train model
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    history = model.fit(
        train_images, train_masks,
        validation_data=(val_images, val_masks),
        batch_size=16,
        epochs=240,
        callbacks=[checkpoint, tensorboard, reduce_lr]
    )
