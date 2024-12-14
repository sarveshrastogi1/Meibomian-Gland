# Meibomian Gland Segmentation

This repository contains the code and resources for the segmentation and extraction of Meibomian glands from medical images. This project uses deep learning techniques to automate the segmentation process, leveraging the power of U-Net and TransUNet architectures. The goal is to accurately identify and segment Meibomian gland regions for medical analysis.

---

## Features
- **U-Net Implementation**: A convolutional neural network for biomedical image segmentation.
- **TransUNet Implementation**: Combines CNNs with transformers for enhanced segmentation performance.
- **Data Preprocessing**: Includes CLAHE (Contrast Limited Adaptive Histogram Equalization) and noise removal techniques to improve image quality.
- **Image Enhancement**: Wallis filter and other methods applied to enhance the visibility of gland structures.
- **Inference Scripts**: Predict Meibomian gland masks for new images.
- **Support for Bounding Boxes**: Detect and crop eyelid regions before segmentation.
- **Metrics and Visualization**: Evaluate the quality of segmentation using standard metrics and visualize results.

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/meibomian-gland-segmentation.git
   cd meibomian-gland-segmentation
   ```

2. **Set Up a Virtual Environment** (Recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## File Structure

```
meibomian-gland-segmentation/
├── models/
│   ├── unet.py        # Training script for U-Net
│   ├── transunet.py   # Training script for TransUNet
│   ├── unet_inference.py       # Inference script for U-Net
│   ├── transunet_inference.py  # Inference script for TransUNet
│
├── data/
│   ├── images/                 # Input images
│   ├── masks/                  # Ground truth masks
│
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
```

---

## How to Run

### 1. Data Preparation
Organize your dataset into the following structure:
```
data/
├── images/   # Raw input images
├── masks/    # Corresponding ground truth segmentation masks
```

Ensure the images and masks are named consistently for pairing during training.

### 2. Training the Models

#### Train U-Net:
```bash
python unet.py
```

#### Train TransUNet:
```bash
python transunet.py
```

### 3. Inference

#### Run U-Net Inference:
```bash
python inference.py
```

#### Run TransUNet Inference:
```bash
python transunet_inference.py
```

---

## Preprocessing and Image Enhancement

1. **CLAHE**: Improves contrast in images to enhance gland visibility.
2. **Wallis Filter**: Sharpens images and focuses on relevant structures.

The preprocessing pipeline is automatically applied during training and inference.

---

## Evaluation Metrics
The segmentation performance is evaluated using the following metrics:
- **Dice Coefficient**: Measures the overlap between predicted and ground truth masks.
- **IoU (Intersection over Union)**: Evaluates the accuracy of the segmentation.
- **Completeness and Correctness**: Specific metrics inspired by building detection tasks.

---

## Results
The project achieves high accuracy in Meibomian gland segmentation using both U-Net and TransUNet. TransUNet demonstrates superior performance, especially in capturing finer details of the gland structures.

---

## Future Work
- Integrating gland extraction for individual analysis.
- Optimizing the model for real-time segmentation.
- Developing an interactive tool for manual adjustments to segmentation masks.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
Special thanks to AutoYOS for providing the dataset and support during the development of this project.

