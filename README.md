# FloodNet Semantic Segmentation Project

This repository contains a complete semantic segmentation pipeline built for the **FloodNet** aerial imagery dataset. The goal is to identify flooded vs non‑flooded objects (buildings, roads, water, vegetation, etc.) using U‑Net architectures in TensorFlow/Keras.

## Problem Description & Dataset

FloodNet is a publicly available challenge dataset consisting of high‑resolution aerial images annotated with **10 semantic classes** (background, flooded/non‑flooded buildings and roads, water, tree, vehicle, pool, grass). The dataset has approximately 398 labeled training pairs. The task is to build a model that performs pixel‑wise classification at **640×640 resolution** (minimum project requirement).

Key dataset details:

* **Track 1** – semantic segmentation
* 10 classes with a highly imbalanced distribution (vehicles and pools are very rare)
* Images are 3‑channel RGB JPGs; masks are PNGs with integer class IDs

For this project the data is downloaded using the `kagglehub` helper, stored under `~/.cache/kagglehub/...` and then split into train/validation/test sets (70/15/15).

## Setup Instructions

1. **Clone the repository** and navigate into it:
	```bash
	git clone https://github.com/Ledja22/mini-project-8.git
	cd mini-project-8
	```
2. **Install dependencies** (preferably in a virtual environment):
	```bash
	pip install -r requirements.txt
	# requirements include tensorflow, albumentations, kagglehub, opencv-python, matplotlib, seaborn, scikit-learn, etc.
	```
3. **Download the FloodNet dataset** by running the first cell in `notebook/notebook.ipynb` or manually via Kaggle credentials. The notebook uses `kagglehub.dataset_download(...)` to fetch the data automatically.
4. **Adjust paths** if necessary: the notebook assumes the dataset lives in `~/.cache/kagglehub/...`; update `DATA_DIR` at the top of the notebook if your environment differs.

## Running the Code

All code is contained in `notebook/notebook.ipynb`. The high‑level workflow is:

1. **Setup & imports** – install packages and configure TensorFlow.
2. **Data exploration & pipeline** – list files, verify image/mask pairs, perform 70/15/15 splits, apply synchronized Albumentations augmentations, and build `tf.data` pipelines.
3. **Model architectures** – define two U‑Net variants:
	* Vanilla U‑Net built from scratch
	* U‑Net with an EfficientNetB0 encoder pretrained on ImageNet (two‑phase training)
4. **Training** – compile and train both models with custom loss functions (Dice, cross‑entropy, or combination) and useful callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard).
5. **Evaluation** – run the test set through both models, compute per‑class IoU/Dice, confusion matrices, and save plots.
6. **Visualization** – display six example predictions (top 3 good, bottom 3 poor) with error maps.

To run end‑to‑end, simply execute the notebook cells in order.

## Results Summary

After training on the provided split (640×640, batch size 4, 25 epochs), the following metrics were obtained on the held‑out test sets.

📊 Summary Table:

| Class               | U‑Net IoU | EffNet IoU | U‑Net Dice | EffNet Dice |
|---------------------|-----------|------------|------------|-------------|
| Background          | 0.0000    | 0.0000     | 0.0000     | 0.0000      |
| Bldg-flooded        | 0.0000    | 0.0000     | 0.0000     | 0.0000      |
| Bldg-intact         | 0.2352    | 0.2582     | 0.3808     | 0.4104      |
| Road-flooded        | 0.0000    | 0.0000     | 0.0000     | 0.0000      |
| Road-intact         | 0.5539    | 0.5579     | 0.7129     | 0.7162      |
| Water               | 0.4632    | 0.4414     | 0.6332     | 0.6124      |
| Tree                | 0.4144    | 0.4333     | 0.5860     | 0.6047      |
| Vehicle             | 0.1018    | 0.0000     | 0.1849     | 0.0000      |
| Pool                | 0.0592    | 0.0000     | 0.1118     | 0.0000      |
| Grass               | 0.7275    | 0.7255     | 0.8423     | 0.8409      |

# U-Net   mIoU=0.2555  mean Dice=0.3452
# EffNet  mIoU=0.2416  mean Dice=0.3185


### Confusion Matrices


## Sample Prediction Visualizations

The notebook also produces a grid of six test images showing:

1. Input image
2. Ground truth mask (color coded)
3. Model prediction mask
4. Error map (grey = correct, red = incorrect)

Three “good” and three “poor” examples are selected based on per‑image mIoU, providing insight into model strengths and weaknesses. The output is saved as `predictions_visualization.png`.

## Team Member Contributions

* Ledja Halltari – dataset handling, data pipeline, augmentation strategy, evaluation metrics
* Ledja Halltari – model architecture design (vanilla U‑Net, EfficientNet backbone), training and hyperparameter tuning
* Ledja Halltari– visualization code, notebook formatting, README documentation, result analysis
* Nicky Cheng - Report and analysis


This README provides the essential information to understand, run, and extend the FloodNet semantic segmentation project. For further experimentation, try altering image resolutions (256 vs 640), changing batch sizes, or swapping in other backbones.
