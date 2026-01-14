
# 🎵 Music Popularity Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a song will be **popular (Hit)** or **not popular (Not Hit)** based on its audio features.
It uses **machine learning classification models** to learn patterns from music data and estimate a song’s success.

The goal is to understand:

* What makes a song popular
* Which features matter the most
* How well a model can predict real-world music performance

---

## 🧠 Problem Statement

Music popularity depends on multiple factors like tempo, energy, danceability, loudness, etc.
Manually predicting success is difficult, so we use **machine learning** to classify songs based on these features.

---

## 🎯 Objective

* Predict song popularity (Hit / Not Hit)
* Analyze important audio features
* Build an interpretable and reliable ML model

---

## 📂 Dataset

* Source: Spotify-style audio features dataset
* Each row represents a song
* Target variable: **Popularity / Hit label**
* Features include:

  * Danceability
  * Energy
  * Loudness
  * Tempo
  * Acousticness
  * Valence
  * Speechiness

---

## ⚙️ Technologies Used

* **Python**
* **Pandas & NumPy** – data handling
* **Matplotlib & Seaborn** – visualization
* **Scikit-learn** – machine learning models

---

## 🧪 Machine Learning Models

* **Random Forest Classifier** 🌲
* (Compared with Decision Tree for understanding improvement)

Why Random Forest?

* Reduces overfitting
* Handles non-linear relationships
* More stable and accurate than a single decision tree

---

## 📊 Model Evaluation

* Accuracy
* Confusion Matrix
* Feature Importance Analysis

Example Output:

```
Prediction: HIT 🎯
Confidence:
HIT: 89.7%
NOT HIT: 10.3%
```

---

## 🔍 Feature Importance

The model identifies which features influence popularity the most, helping answer:

* What audio characteristics increase hit potential?
* Which features contribute the least?

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone <repository-link>
```

2. Open the notebook

```bash
Music_popularity_prediction.ipynb
```

3. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run all cells step by step

---

## 📈 Results

* The model successfully learns patterns in music data
* Random Forest performs better than a single Decision Tree
* Feature importance provides meaningful insights

---
## 🚀 Live Demo
👉 https://musicpopularitypredictionmodel-lh8gocrdgsdcnavxyknrdh.streamlit.app/


## 🔮 Future Improvements

* Use regression to predict exact popularity score
* Try XGBoost or Gradient Boosting
* Add genre and artist-level features
* Deploy as a web app using Streamlit

---

