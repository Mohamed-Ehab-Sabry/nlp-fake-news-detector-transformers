# 🔍 Sentiment & Fake News Detection Pipeline

An end-to-end NLP pipeline for **sentiment analysis** and **fake news detection**, progressing from classical ML to transformers and explainable AI.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

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
This project builds a multi-phase NLP pipeline that:
1. Detects **sentiment polarity** from social media and reviews
2. Classifies **news authenticity** (real vs. fake)

## 📊 Datasets
| Dataset | Purpose | Source |
|---------|---------|--------|
| Sentiment140 | Twitter sentiment analysis | [Link](http://help.sentiment140.com/for-students) |
| LIAR | Fake news classification | [Link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) |
| Amazon Reviews | Product sentiment analysis | [Link](https://www.kaggle.com/bittlingmayer/amazonreviews) |

> ⚠️ Datasets are **not included** in the repo. See `data/README.md` for download instructions.

## 🔬 Project Phases

### Phase 1 — Classical ML
- Text EDA (word clouds, frequency plots)
- TF-IDF + KNN, Naïve Bayes, Random Forest, XGBoost
- Dimensionality reduction (TruncatedSVD/LSA)

### Phase 2 — Deep Learning
- 1D-CNN on word embeddings
- Autoencoder for anomalous text detection
- Transfer learning (GloVe, FastText)

### Phase 3 — Transformers & XAI
- LSTM/Bi-LSTM with Attention
- BERT/DistilBERT fine-tuning
- SHAP & LIME for interpretability
- Conditional GAN for adversarial samples

## ⚙️ Installation
```bash
git clone https://github.com/Mohamed-Ehab-Sabry/sentiment-fakenews-detection.git
cd sentiment-fakenews-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🤝 Contributing
Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## 👥 Team

| Name | Role | Responsibilities | GitHub |
|------|------|-----------------|--------|
| Sama | Data Collection & Cleaning | Prepare datasets; produce clean text data ready for analysis and modeling | [@Sam-Gyu](https://github.com/Sam-Gyu) |
| Yasmin | Text EDA (Exploratory Data Analysis) | Analyze word distributions and text patterns; produce visual insights about the text data | [@yasmin2027](https://github.com/yasmin2027) |
| Merehan | Feature Engineering (Text → Numerical Features) | Convert text into numerical features (TF-IDF, embeddings) that ML models can use | [@merehan123](https://github.com/merehan123) |
| Mohamed Ehab | Classical ML Models | Train KNN, Naïve Bayes, Random Forest, and XGBoost classifiers on TF-IDF features | [@Mohamed-Ehab-Sabry](https://github.com/Mohamed-Ehab-Sabry) |
| Ayman | Model Evaluation & Optimization | Evaluate model performance; apply dimensionality reduction and hyperparameter tuning | [@ayman-n1](https://github.com/ayman-n1) |

## 📜 License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
