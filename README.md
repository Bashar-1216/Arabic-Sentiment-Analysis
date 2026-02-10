<![CDATA[<div align="center">

# 🇸🇦 Arabic Sentiment Analysis Using Deep Learning 🧠

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

> 🔬 مشروع تحليل المشاعر في النصوص العربية باستخدام التعلم العميق

**A comprehensive Arabic Sentiment Analysis project comparing BiLSTM with CAMeL-BERT embeddings against a Fine-tuned CAMeL-BERT model.**

---

</div>

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [📂 Project Structure](#-project-structure)
- [📊 Datasets](#-datasets)
- [🧹 Text Preprocessing](#-text-preprocessing)
- [🏗️ Model Architectures](#️-model-architectures)
- [📈 Results](#-results)
- [⚙️ Requirements](#️-requirements)
- [🚀 How to Run](#-how-to-run)
- [✨ Key Features](#-key-features)
- [🛠️ Technologies Used](#️-technologies-used)

---

## 🌟 Overview

This project implements and compares **two powerful deep learning approaches** for Arabic Sentiment Analysis:

| # | Approach | Description |
|:-:|----------|-------------|
| 1️⃣ | **BiLSTM + CAMeL-BERT** | A Bidirectional LSTM model leveraging pre-trained word embeddings from CAMeL-BERT |
| 2️⃣ | **Fine-tuned CAMeL-BERT (LLM)** | Direct fine-tuning of the CAMeL-BERT model for sequence classification |

> 🎯 Both models are trained on the **AraSenti** dataset and evaluated on an out-of-domain **HIA Qatar Airport Tweets** dataset to assess **cross-domain generalization** capability.

---

## 📂 Project Structure

```

├── 📓 Arabic-Sentiment-Analysis.ipynb          # Main Jupyter Notebook (training, evaluation, comparison)
├── 📊 AraSenti_all.xlsx          # Training dataset (~15,751 samples)
├── 📊 HIAQatar_tweets.xlsx       # Testing dataset (~151 samples)
└── 📝 README.md                  # This file
```

---

## 📊 Datasets

| Dataset | Description | Samples | Labels |
|:-------:|:-----------:|:-------:|:------:|
| 📗 **AraSenti** | Multi-source Arabic sentiment corpus | 15,751 | Negative (0), Positive (1), Neutral (2) |
| 📘 **HIA Qatar Tweets** | Airport-related Arabic tweets | 151 | Negative (0), Positive (1), Neutral (2) |

> ⚠️ **Note:** The model is trained on AraSenti and tested on HIA Qatar Tweets to measure **cross-domain** performance.

---

## 🧹 Text Preprocessing

Arabic text is cleaned using the `clean_tweet()` function which performs the following pipeline:

| Step | Operation | Description |
|:----:|-----------|-------------|
| 1️⃣ | 🔗 URL Removal | Remove `http`, `www`, `https` links |
| 2️⃣ | 📛 Mention/Hashtag Removal | Remove `@` and `#` tags |
| 3️⃣ | 🔤 English Removal | Remove English characters & digits |
| 4️⃣ | ➖ Underscore Removal | Remove underscore characters |
| 5️⃣ | 🔁 Repeated Char Reduction | Normalize repeated characters |
| 6️⃣ | 🔄 Arabic Normalization | `إأآا` → `ا`, `ة` → `ه`, `ى` → `ي` |
| 7️⃣ | 🧹 Non-Arabic Removal | Keep only Arabic text & whitespace |
| 8️⃣ | 📐 Whitespace Normalization | Trim & normalize spaces |

---

## 🏗️ Model Architectures

### 1️⃣ BiLSTM with CAMeL-BERT Embeddings

```
📥 Input Text
    ↓
🔤 CAMeL-BERT Tokenizer (max_length=60)
    ↓
🧠 CAMeL-BERT Encoder → Word Embeddings (768-dim)
    ↓
🔁 Bidirectional LSTM (hidden_dim=128, 1 layer)
    ↓
💧 Dropout (0.5)
    ↓
🔢 Linear Layer (256 → 3 classes)
    ↓
📊 Output: Negative / Positive / Neutral
```

| ⚙️ Component | 📝 Details |
|:-------------:|:----------:|
| Embedding | CAMeL-BERT (`last_hidden_state`, 768-dim) |
| LSTM | Bidirectional, hidden_dim=128, 1 layer |
| Dropout | 0.5 |
| Classifier | Linear (256 → 3 classes) |
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Early Stopping | patience=5, based on validation loss |

---

### 2️⃣ Fine-tuned CAMeL-BERT (LLM)

```
📥 Input Text
    ↓
🔤 CAMeL-BERT Tokenizer (max_length=60)
    ↓
🧠 CAMeL-BERT for Sequence Classification (Full Fine-tuning)
    ↓
🔍 Grid Search: lr ∈ {2e-5, 3e-5} × epochs ∈ {3, 4}
    ↓
📊 Output: Negative / Positive / Neutral
```

| ⚙️ Component | 📝 Details |
|:-------------:|:----------:|
| Base Model | `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` |
| Task | Sequence Classification (3 classes) |
| Hyperparameter Search | lr ∈ {2e-5, 3e-5}, epochs ∈ {3, 4} |
| Batch Size | 16 |
| Weight Decay | 0.01 |
| Label Smoothing | 0.1 |
| Best Model Selection | Based on validation F1 (macro) |

---

## 📈 Results

### 🏆 Test Set Performance (HIA Qatar Airport Tweets)

<div align="center">

| 🤖 Model | 🎯 Accuracy | 📊 F1 Score (Macro) |
|:---------:|:----------:|:------------------:|
| **BiLSTM + CAMeL-BERT** | **0.7616** ✅ | **0.7464** ✅ |
| **Fine-tuned LLM** | 0.7550 | 0.7407 |

</div>

### 📋 BiLSTM — Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 😡 Negative | 0.94 | 0.72 | 0.81 | 82 |
| 😊 Positive | 0.82 | 0.80 | 0.81 | 40 |
| 😐 Neutral | 0.49 | 0.83 | 0.62 | 29 |

### 📋 Fine-tuned LLM — Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 😡 Negative | 0.97 | 0.68 | 0.80 | 82 |
| 😊 Positive | 0.68 | 0.85 | 0.76 | 40 |
| 😐 Neutral | 0.56 | 0.83 | 0.67 | 29 |

---

## ⚙️ Requirements

```txt
torch
transformers
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone <repository-url>
cd Arabic-Sentiment-Analysis
```

### 2. Install dependencies
```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### 3. Ensure GPU availability ⚡
```python
import torch
print(torch.cuda.is_available())  # Should be True ✅
```

### 4. Run the notebook 📓
- Open `Arabic-Sentiment-Analysis.ipynb` in **Jupyter Notebook** or **Kaggle**
- Execute all cells sequentially

> 💡 **Tip:** This project was originally developed and run on **Kaggle** with GPU acceleration. Dataset paths may need to be adjusted if running locally.

---

## ✨ Key Features

| Feature | Description |
|:-------:|:-----------:|
| 🔒 **Reproducibility** | Random seed fixed at `42` for deterministic results |
| 🇸🇦 **Arabic-specific** | Comprehensive text normalization pipeline for Arabic |
| 🔄 **Cross-domain** | Training on AraSenti, testing on airport tweets |
| 🔍 **Hyperparameter Search** | Grid search over learning rates and epochs |
| ⚖️ **Model Comparison** | Side-by-side evaluation with confusion matrices |
| 📊 **Visualization** | Training curves & confusion matrix heatmaps |

---

## 🛠️ Technologies Used

<div align="center">

| Technology | Purpose |
|:----------:|:-------:|
| 🐍 **Python 3** | Programming Language |
| 🔥 **PyTorch** | Deep Learning Framework |
| 🤗 **Hugging Face Transformers** | Pre-trained Models & Fine-tuning |
| 🐪 **CAMeL-BERT** | Arabic Language Model |
| 📊 **scikit-learn** | Evaluation Metrics |
| 📈 **Matplotlib & Seaborn** | Data Visualization |

</div>

---

<div align="center">

### 📜 License

This project is developed for **academic purposes** as part of the **Natural Language Processing** course.

---

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>
]]>
