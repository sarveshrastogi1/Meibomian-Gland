import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K


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





def preprocess_data(image_folder, mask_folder):
    image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder)])
    mask_files = sorted([os.path.join(mask_folder, file) for file in os.listdir(mask_folder)])

    image_dict = {os.path.splitext(os.path.basename(file))[0]: file for file in image_files}
    mask_dict = {os.path.splitext(os.path.basename(file))[0]: file for file in mask_files}

    common_keys = set(image_dict.keys()).intersection(mask_dict.keys())

    images = [cv2.imread(image_dict[key], cv2.IMREAD_GRAYSCALE) for key in common_keys]
    masks = [cv2.imread(mask_dict[key], cv2.IMREAD_GRAYSCALE) for key in common_keys]

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images = [clahe.apply(img) for img in images]


    images = [cv2.resize(img, (256, 256)) for img in images]
    masks = [cv2.resize(mask, (256, 256)) for mask in masks]

    images = np.array(images).reshape(-1, 256, 256, 1)
    masks = np.array(masks).reshape(-1, 256, 256, 1)

    images = images / 255.0
    masks = masks / 255.0
    return images, masks


# Load and preprocess training and validation data
train_images, train_masks = preprocess_data('new_data1/train/images', 'new_data1/train/mask')
val_images, val_masks = preprocess_data('new_data1/val/images', 'new_data1/val/mask')


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def combined_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def unet_model(input_size=(256, 256, 1)):
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

    # Decoder Path
    u6 = UpSampling2D((2, 2))(c5)
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
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coefficient, iou_score])
    return model

model = unet_model()

checkpoint = ModelCheckpoint('logs_new2\\unet_best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir='logs_new2', histogram_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_lr=1e-7)


history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    batch_size=16,
    epochs=240,
    callbacks=[checkpoint, tensorboard, reduce_lr]
)
