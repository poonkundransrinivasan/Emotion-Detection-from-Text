# Emotion Detection from Text

This repository contains the code for a multi-label emotion classification model that identifies seven emotions from English text: **admiration**, **amusement**, **gratitude**, **love**, **pride**, **relief**, and **remorse**.

## 🔍 Overview

The goal is to detect one or more emotions present in a given sentence. A transformer-based architecture is used to capture subtle emotional expressions in natural language effectively.

## 🧠 Model Architecture

- **Base Model**: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased), frozen during training for efficiency
- **Custom Classifier**:
  - Bidirectional GRUs
  - Dropout layers for regularization
  - Dense layers with sigmoid activation for multi-label output
- **Frameworks**:
  - Hugging Face Transformers & Datasets
  - TensorFlow / Keras

## ⚙️ Features

- Mixed precision training for GPU speedup
- EarlyStopping and ModelCheckpoint support
- Command-line interface for easy training and inference
- Full preprocessing and prediction pipeline

## 📁 Project Structure

```
├── data/                   # Data loading and preprocessing
├── models/                 # Model architecture and weights
├── code.py                 # Main script for training and prediction
├── utils.py                # Utility functions
├── requirements.txt        # Required Python packages
```

## 🚀 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/poonkundransrinivasan/Emotion-Detection-from-Text.git
cd Emotion-Detection-from-Text
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python code.py train
```

### 4. Run Predictions

```bash
python code.py predict
```

> The script uses `argparse` to route between `train()` and `predict()` based on the command-line argument provided.

## 🧪 Future Improvements

- Add more emotion categories
- Support additional languages
- Deploy as a REST API
