import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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


def combined_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess a single image for prediction with CLAHE."""
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_size = original_image.shape

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(original_image)

    # Resize and normalize
    image = cv2.resize(image_clahe, target_size)
    image = image.reshape(1, target_size[0], target_size[1], 1)
    image = image / 255.0
    return image, original_size


def postprocess_mask(mask, original_size, target_size=(256, 256)):
    """Postprocess the predicted mask with additional post-processing steps."""
    mask = (mask > 0.5).astype(np.uint8)
    mask = mask[0, :, :, 0]

    # Resize mask to target size
    mask = cv2.resize(mask, target_size)

    # Apply morphological operations (closing and opening) to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Optionally apply Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Resize mask back to original image size
    mask = cv2.resize(mask, (original_size[1], original_size[0]))
    return mask


def draw_largest_contour_on_image(image_path, mask, original_size, output_path):
    """Draw the largest contour of the mask on the original image."""
    original_image = cv2.imread(image_path)

    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, just return the original image
    if not contours:
        cv2.imwrite(output_path, original_image)
        print(f"No contours found. Saved original image to {output_path}")
        return

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the original image
    cv2.drawContours(original_image, [largest_contour], -1, (0, 255, 0), 2)  # Draw green contour

    cv2.imwrite(output_path, original_image)
    print(f"Saved image with the largest contour to {output_path}")


def predict_mask(model, image_path, output_path):
    """Predict and save the image with the largest mask contour."""
    image, original_size = preprocess_image(image_path)
    prediction = model.predict(image)
    mask = postprocess_mask(prediction, original_size)
    draw_largest_contour_on_image(image_path, mask, original_size, output_path)


def main(input_folder, output_folder, model_path):
    # Load the trained model
    model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient,
                                                   'iou_score': iou_score})
    print("Model loaded successfully.")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the input folder and predict the masks
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            predict_mask(model, image_path, output_path)
    print("Inference completed.")


if __name__ == "__main__":
    input_folder = 'C:\\Users\\rasto\\Downloads\\test_images_to_sarvesh\\test_images_to_sarvesh'
    output_folder = 'C:\\Users\\rasto\\Downloads\\output_new'


    model_path = 'C:\\Users\\rasto\\Downloads\\pythonproject\\unet\\logs_new3\\unet_best_model.keras'
    main(input_folder, output_folder, model_path)
