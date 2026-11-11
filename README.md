# Homework 1: Image Enhancement
**Computer Vision - Fall 2026**

This repository contains a classical image processing pipeline designed to enhance color images affected by low contrast, non-uniform illumination, and noise. The project evaluates the trade-offs of different filtering techniques without relying on deep learning models.

## Repository Structure
```text
.
├── src/
│   ├── main.py            # Main script to run the pipeline
│   └── functions.py       # Helper functions (metrics, CSV builder, album visualization)
├── samples/               # Directory containing the 5 original input images
│   ├── lc_pd.jpg
│   ├── bar.jpg
│   ├── pd1.jpg
│   ├── mtty.jpg
│   └── altare.jpg
├── results/               # Generated automatically: contains side-by-side comparison images
├── results.csv            # Generated automatically: quantitative metrics (PSNR, SSIM)
└── README.md              # This file
```

## Dependencies and Requirements
- **Python version**: 3.10 or higher recommended.
- **Libraries**:
  - OpenCV (`cv2`)
  - NumPy (`numpy`)
  - scikit-image (`skimage`)
  - Pillow (`PIL`)

You can install all required dependencies using `pip` by running the following command in your terminal:
```bash
pip install opencv-python numpy scikit-image pillow
```

## How to Run the Pipeline
To reproduce the experiment, generate the comparison images, and compute the metrics, run the `main.py` from the root directory of this repository:
```bash
python3 src/main.py
```