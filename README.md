# Machine Learning Model to Predict Angle and Speed for a Self-Driving Pi Car

This project uses **Transfer Learning with MobileNetV3-Small** to predict the **steering angle** and **speed** of a self-driving Raspberry Pi car from camera images.

The model was trained using **PyTorch** and leverages a pretrained MobileNet backbone to efficiently learn visual driving patterns from image data.

---

## Project Overview

The goal of this project is to build a computer vision regression model capable of predicting:

- Steering Angle
- Vehicle Speed

from road images captured by the Pi Car.

Instead of training a CNN from scratch, this implementation uses **MobileNetV3-Small pretrained on ImageNet** and fine-tunes it for regression.

---

## Features

- Transfer Learning with MobileNetV3-Small
- PyTorch implementation
- Mixed Precision Training (AMP)
- Data Augmentation
- Early Stopping
- Cosine Annealing Learning Rate Scheduler
- Automatic corrupted image filtering
- GPU acceleration support
- CSV submission generation

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- PIL
- Google Colab

---

## Dataset Structure

Expected dataset structure:

NB: Training Data not provided, contact me if needed

```bash
training_data/
│
├── training_data/
│   ├── 1.png
│   ├── 2.png
│   └── ...
│
test_data/
│
├── test_data/
│   ├── 1.png
│   ├── 2.png
│   └── ...

train.csv

The train.csv file should contain:
image_id,angle,speed
1,0.12,0.45
2,-0.08,0.39

