# 🔍 NLP Fake News Detection: From Classical ML to Transformers

A comprehensive, multi-phase NLP pipeline for **fake news classification**, progressing from classical machine learning to state-of-the-art transformer models with 89% accuracy.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

## 📖 Table of Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Project Phases](#project-phases)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

## 🧠 Overview

This project demonstrates a comprehensive progression from classical machine learning to transformer-based models for fake news detection. Through three phases, we systematically improve classification accuracy from **70% to 89%** using increasingly sophisticated architectures.

**Key Results**:
| Phase | Best Model | Accuracy | F1-Score | AUC-ROC |
|-------|-----------|----------|----------|---------|
| **1** | Naive Bayes | 0.70 | 0.68 | 0.72 |
| **2** | GloVe+BiLSTM | 0.77 | 0.75 | 0.81 |
| **3** | BERT Ensemble | **0.89** | **0.88** | **0.93** |

## 📊 Dataset

**Twitter Fake News Dataset**
- **Size**: 1,482,190 samples
- **Split**: 80% training, 20% testing
- **Classes**: Binary (Real vs. Fake news)
- **Features**: Raw text (140-150 characters average)
- **Class Balance**: ~60% real, ~40% fake

> Note: Dataset not included in repo. Instructions for dataset preparation available in `data/README.md`

## 🔬 Project Phases

### Phase 1 — Classical Machine Learning (Baseline)
- **EDA**: Word clouds, frequency distributions, statistics
- **Models**: KNN, Naive Bayes, Random Forest, XGBoost
- **Features**: TF-IDF vectorization (10,000 features)
- **Optimization**: Dimensionality reduction (SVD), hyperparameter tuning
- **Report**: [Phase_1_Report.md](docs/Phase_1_Report.md)
- **Notebooks**: `notebooks/Phase_1/`

### Phase 2 — Deep Learning Models
- **CNN**: 1D Convolutional networks with word embeddings
- **Autoencoder**: Unsupervised anomaly detection
- **Transfer Learning**: Pre-trained embeddings (GloVe, FastText, Word2Vec)
- **Optimizer Study**: Comparison of SGD, Adam, RMSprop, AdaGrad
- **Report**: [Phase_2_Report.md](docs/Phase_2_Report.md)
- **Notebooks**: `notebooks/Phase_2/`

### Phase 3 — Transformer Models & Ensemble
- **RNN**: Basic recurrent networks
- **GRU**: Gated recurrent units (improved RNN)
- **LSTM**: Long short-term memory networks
- **BERT**: Fine-tuned transformer for state-of-the-art performance
- **Ensemble**: Voting ensemble combining BERT + LSTM + GRU
- **Report**: [Final_Report.md](docs/Final_Report.md)
- **Notebooks**: `notebooks/Phase_3/`

## 📁 Project Structure

```
nlp-fake-news-detector-transformers/
├── notebooks/
│   ├── Phase_1/                    # Classical ML notebooks
│   │   ├── 01_EDA.ipynb
│   │   ├── 01_KNN-Naive_Bayes.ipynb
│   │   ├── 01_feature_engineering_ml.ipynb
│   │   └── 01_XGBoost_Random-Forest.ipynb
│   ├── Phase_2/                    # Deep learning notebooks
│   │   ├── 02_CNN.ipynb
│   │   ├── 02-Autoencoder.ipynb
│   │   ├── 02-Transfer_Learning.ipynb
│   │   ├── 02-CNN_Optimization.ipynb
│   │   └── 02-feature_engineering_dl.ipynb
│   └── Phase_3/                    # Transformer notebooks
│       ├── 03_RNN.ipynb
│       ├── 03_GRU.ipynb
│       ├── 03_LSTM.ipynb
│       ├── 03-BERT_data_prep.ipynb
│       └── 03-BERT.ipynb
├── src/
│   ├── data/                       # Data loading & preprocessing
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── bert_preprocessor.py
│   ├── features/                   # Feature engineering
│   │   ├── build_features.py
│   │   └── dl_text_preprocessing.py
│   └── models/                     # Trained model modules
│       ├── __init__.py
│       ├── classical_models.py     # KNN, NB, RF, XGBoost
│       ├── deep_learning.py        # CNN, Autoencoder, BiLSTM
│       ├── transfer_learning.py    # Pre-trained embeddings
│       └── transformer_models.py   # BERT, Ensemble
├── docs/
│   ├── Phase_1_Report.md           # Phase 1 detailed report
│   ├── Phase_2_Report.md           # Phase 2 detailed report
│   └── Final_Report.md             # Comprehensive final report
├── results/
│   ├── figures/                    # Visualizations & plots
│   └── models/                     # Trained model artifacts
├── data/                           # Datasets (not included)
│   ├── raw/
│   ├── processed/
│   ├── dl_features/
│   └── saved_features/
├── configs/                        # Configuration files
├── requirements.txt
├── README.md
└── LICENSE
```

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/Mohamed-Ehab-Sabry/nlp-fake-news-detector-transformers.git
cd nlp-fake-news-detector-transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

```
tensorflow==2.10+
transformers==4.20+
scikit-learn==1.0+
pandas==1.3+
numpy==1.21+
matplotlib==3.4+
seaborn==0.11+
nltk==3.6+
joblib==1.0+
xgboost==1.5+
```

## 🚀 Quick Start

### Running Phase 3 (BERT) Model

```python
from src.models import BertClassifier

# Initialize BERT classifier
model = BertClassifier(model_name="bert-base-uncased", max_length=150)

# Prepare your texts
texts = ["Your news text here", "Another news text"]
predictions, probabilities = model.predict(texts)
```

### Using Pre-trained Models

```python
from src.models import get_classical_model, get_dl_model

# Load classical model
knn = get_classical_model("knn_model.joblib")

# Load deep learning model
cnn = get_dl_model("cnn_model.keras")

# Make predictions
predictions = model.predict(features)
```

### Viewing Reports

- **Phase 1 Analysis**: [Phase_1_Report.md](docs/Phase_1_Report.md)
- **Phase 2 Experiments**: [Phase_2_Report.md](docs/Phase_2_Report.md)
- **Complete Project**: [Final_Report.md](docs/Final_Report.md)

## 📊 Performance Benchmarks

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | Training Time |
|-------|----------|-----------|--------|-----|---------------|
| Naive Bayes | 0.70 | 0.68 | 0.68 | 0.68 | <1 min |
| KNN | 0.68 | 0.68 | 0.69 | 0.68 | 5 min |
| Random Forest | 0.71 | 0.69 | 0.70 | 0.69 | 3 min |
| XGBoost | 0.72 | 0.70 | 0.71 | 0.70 | 5 min |
| CNN | 0.75 | 0.72 | 0.74 | 0.73 | 4 min |
| GRU | 0.80 | 0.81 | 0.79 | 0.80 | 7 min |
| LSTM | 0.82 | 0.84 | 0.78 | 0.80 | 12 min |
| BERT | 0.87 | 0.88 | 0.85 | 0.86 | 20 min |
| **Ensemble** | **0.89** | **0.90** | **0.87** | **0.88** | 25 min |

### Key Improvements

- **Phase 1 → Phase 2**: +7% accuracy (Deep learning advantage)
- **Phase 2 → Phase 3**: +12% accuracy (Transformer advantage)
- **Total Gain**: +19% from baseline to final ensemble

## 🤝 Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## 👥 Team

| Name | Role | Responsibilities | GitHub |
|------|------|-----------------|--------|
| Sama | Data Collection & Cleaning | Dataset preparation and preprocessing | [@Sam-Gyu](https://github.com/Sam-Gyu) |
| Yasmin | Text EDA & Analysis | Exploratory data analysis with visualizations | [@yasmin2027](https://github.com/yasmin2027) |
| Merehan | Feature Engineering | Text-to-numerical feature extraction | [@merehan123](https://github.com/merehan123) |
| Mohamed Ehab | Classical ML & Orchestration | Phase 1 models and project coordination | [@Mohamed-Ehab-Sabry](https://github.com/Mohamed-Ehab-Sabry) |
| Ayman | Model Evaluation & Deep Learning | Phase 2-3 models and comprehensive evaluation | [@ayman-n1](https://github.com/ayman-n1) |

## 📜 License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
