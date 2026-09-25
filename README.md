# Comparative Evaluation of Machine Learning and Deep Learning Models for Unseen-Family DGA Domain Detection

A machine learning and deep learning based system for detecting Domain Generation Algorithm (DGA) generated domains and evaluating how well different models generalize to previously unseen DGA families.

## Project Overview

Domain Generation Algorithms (DGAs) are commonly used by malware to generate large numbers of domain names for communication with Command-and-Control (C&C) servers. Since newly generated domains may not appear in existing blacklists, detecting them from their domain strings is an important cybersecurity problem.

This project performs a comparative evaluation of machine learning and deep learning models under a **family-disjoint unseen-family evaluation setting**. Five DGA families are completely excluded from training and validation and are used only for the final test:

- Nymaim
- Pykspa
- Qadars
- Ramnit
- Shifu

The main purpose is to evaluate how well different domain representations and models generalize to DGA families that were not seen during model development.

## Dataset

The experiments use the **ARGENCON DGA Detection Dataset** from the `harpomaxx/dga-detection` repository on Hugging Face.

- Total domains: **2,918,496**
- Benign domains: **1,003,161**
- DGA-generated domains: **1,915,335**
- DGA families: **51**

The data is cleaned by removing missing and empty values, trimming whitespace, identifying DGA families, converting labels into binary classes, and removing duplicate domains.

## Experimental Setup

The dataset is divided using a family-disjoint evaluation strategy.

| Dataset | Samples |
|---|---:|
| Training | 1,932,263 |
| Validation | 340,988 |
| Unseen-Family Test | 284,168 |

The final unseen-family test set contains:

- 142,084 DGA domains
- 142,084 benign domains

The five selected DGA families are completely excluded from training and validation.

## Domain Representations

Three main representations are used:

### 1. Lexical Features

Ten lexical characteristics are extracted from each domain:

- Domain length
- Digit count
- Digit ratio
- Vowel ratio
- Consonant ratio
- Maximum consecutive consonants
- Maximum consecutive digits
- Shannon entropy
- Special-character count
- Special-character ratio

### 2. Character-Level TF-IDF

Character n-grams from **2-gram to 5-gram** are used with a maximum feature space of **300,000 dimensions**.

### 3. Character Sequence

Domain strings are encoded as character sequences and used by character-level deep learning models.

## Models

Six main models are evaluated:

1. Logistic Regression + Character TF-IDF
2. Random Forest + Lexical Features
3. XGBoost + Lexical Features
4. Lexical Neural Network
5. Character CNN
6. Hybrid CNN + Lexical

An additional ensemble combines:

- Logistic Regression
- Weighted Hybrid CNN + Lexical

## Performance

All models are evaluated on the same unseen-family test set using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)

### Comparative Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 81.84% | 87.36% | 74.45% | 80.39% | 0.9170 | 0.6439 |
| XGBoost | 81.77% | 78.11% | 88.28% | 82.88% | 0.9218 | 0.6409 |
| Lexical Neural Network | 83.87% | 90.03% | 76.18% | 82.53% | 0.9267 | 0.6856 |
| Logistic Regression + TF-IDF | 95.12% | 98.16% | 91.97% | 94.96% | 0.9922 | 0.9042 |
| Character CNN | 95.86% | 98.25% | 93.39% | 95.76% | 0.9938 | 0.9184 |
| Weighted Hybrid CNN + Lexical (threshold = 0.50) | 96.03% | 98.73% | 93.25% | 95.91% | 0.9945 | 0.9220 |
| LR + Weighted Hybrid CNN Ensemble | 96.20% | 99.01% | 93.33% | 96.09% | 0.9952 | 0.9256 |
| Weighted Hybrid CNN + Lexical (threshold = 0.20) | 96.57% | 97.12% | 95.98% | 96.55% | 0.9945 | 0.9314 |

The validation-optimized Hybrid CNN + Lexical model achieved 96.57% accuracy, 95.98% recall, 96.55% F1-score and 0.9314 MCC at a decision threshold of 0.20. The Logistic Regression and Weighted Hybrid CNN ensemble achieved 99.01% precision and 0.9952 ROC-AUC.

## System Interface

A user-facing DGA Domain Detector interface was developed for domain-level analysis.

The interface provides:

- Domain prediction
- DGA score
- Confidence analysis
- Model agreement
- Individual model predictions
- Domain feature inspection
- Model comparison
- Hybrid model analysis

The system allows a user to enter a domain name and view predictions from the trained machine learning and deep learning models.

## Repository Structure

```text
DGA-Malicious-Domain-Detection/
│
├── app.py
│
├── features/
│   ├── char_tfidf_vectorizer.pkl
│   ├── character_tokenizer.json
│   └── deployment_config.json
│
├── models/
│   ├── character_cnn_only_best.keras
│   ├── hybrid_cnn_weighted_best.keras
│   ├── lexical_only_best.keras
│   ├── logistic_tfidf.pkl
│   ├── random_forest_lexical.pkl
│   └── xgboost_lexical.pkl
│
├── templates/
│   └── index.html
│
└── .gitattributes
