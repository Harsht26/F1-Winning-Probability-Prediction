# 🏎️ F1 Race Winning Probability Prediction (Random Forest)

## 📌 Project Overview
This project estimates the probability of a Formula 1 driver winning an upcoming race using historical race performance data.

Instead of directly predicting the winner, the model calculates **winning probabilities** using a Random Forest classifier (`predict_proba()`), allowing a more realistic and interpretable output.

---

## 📊 Dataset
The dataset includes historical F1 race data such as:

- Driver
- Team
- Grid Position
- Points
- Race Results
- Other race performance metrics

A target column `winner` was created:
- `1` → Driver finished in Position 1  
- `0` → Otherwise  

---

## ⚙️ Data Preprocessing

- One Hot Encoding for categorical features
- Train-Test Split (80-20)
- Class imbalance handling using **upsampling**
- Feature selection
- Custom probability threshold (15%) for classification



## 🤖 Model Used

- **Random Forest Classifier**
  - `n_estimators=100`
  - `class_weight='balanced'`
  - `random_state=42`

The model predicts probabilities using:
```python
model.predict_proba()

## Model Performance
Accuracy: 100%
F1 Score: High
Precision & Recall evaluated
Confusion Matrix generated



##Why 100% Accuracy?
(1) - The model achieved 100% accuracy on the test set.However, this is mainly due to:
(2)-Small dataset size
(3)-Limited variation in race data
(4)-Possible strong patterns in historical results
(5)-Train-test split on limited samples

```python
model.predict_proba()
