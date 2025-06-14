# 💳 UPI Transaction Fraud Detection

This project is a machine learning web application built using **Streamlit** that detects fraudulent UPI (Unified Payments Interface) transactions. The model is trained using real-world-inspired features and balanced using **SMOTE** to handle class imbalance.

---

## 🔍 Features

- **Manual Transaction Check**: Enter transaction details one-by-one and check if it's fraudulent.
- **Bulk CSV Upload**: Upload a CSV file of transactions and get fraud predictions for each.
- **Model Accuracy**: Cross-validated accuracy over 90%.
- **Balanced Dataset**: SMOTE used to oversample the minority (fraud) class.
- **XGBoost Classifier**: Lightweight and fast model with high precision and recall.

---

🛠 Tech Stack
Python

Streamlit

Pandas / NumPy

Scikit-learn

XGBoost

imbalanced-learn (SMOTE)

Pickle / Joblib for model persistence

---
📦 Project Structure
```bash
📁 upi-fraud-detector/
│
├── app.py                   # Streamlit app
├── model.pkl                # Trained XGBoost model (pickle file)
├── model_features.pkl       # Stored feature names from training
├── Sample_DATA.csv          # Sample input CSV
├── requirements.txt         # Python dependencies
└── README.md                # You're reading this!
```

🚀 Getting Started
1. Clone the repository
```bash
   git clone https://github.com/Qudsiya954/Fraud-Detection.git
   cd Fraud-Detection
```
2. Install dependencies
```bash
pip install -r requirements.txt
````

🧠 Model Details
Algorithm: XGBoost Classifier

Class Balancing: SMOTE (Synthetic Minority Oversampling Technique)

Evaluation Metrics:

Accuracy: ~97%

Precision/Recall (Fraud): ~96-98%

F1 Score: ~97%

📌 To Do
Add more feature engineering

Deploy to cloud (e.g., Streamlit Cloud, Heroku)

Add unit tests

👩‍💻 Developed By
Qudsiya Siddique
