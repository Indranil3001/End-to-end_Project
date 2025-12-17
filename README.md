# End-to-end_Project
Spam Detection Classifier
SMS Spam Detection Using Machine Learning

A university project focused on building and evaluating machine learning models to classify SMS messages as spam or ham (not spam).
The project compares multiple algorithms, performs exploratory data analysis (EDA), processes text data, and identifies the most effective model for spam detection.

Project Overview

This project aims to detect spam in SMS messages using supervised machine learning techniques.
The goal is to build a model that can accurately classify messages and minimize false positives—ensuring important legitimate messages are not incorrectly flagged as spam.

📊 Dataset

The dataset used is the SMS Spam Collection Dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data

It contains 5,574 SMS messages, each labeled as either:

ham – legitimate message

spam – unsolicited or malicious message

Models Used

The following machine learning algorithms were trained and evaluated:

Multinomial Naive Bayes

Bernoulli Naive Bayes

Gaussian Naive Bayes

K-Nearest Neighbors (KNN)

Support Vector Classifier (SVC)

Logistic Regression

Best Performing Model: Bernoulli Naive Bayes

Bernoulli NB achieved perfect precision (1.0) and the highest accuracy among all tested models, making it the best fit for spam detection where false positives must be minimized.

📁 Project Structure
├── spam_detection.ipynb     # Main notebook containing EDA, preprocessing, model training and evaluation
├── data.csv                 # Dataset used for the analysis and model building
└── README.md                # Project documentation

## 📈 **Results**

Model Performance Summary  
(Precision — Accuracy — F1 Score)

| Model                     | Precision | Accuracy | F1 Score |
|--------------------------|-----------|----------|----------|
| Multinomial Naive Bayes  | 1.000000  | 0.958414 | 0.815451 |
| **Bernoulli Naive Bayes** | **1.000000** | **0.965184** | **0.850000** |
| K-Nearest Neighbors      | 1.000000  | 0.905222 | 0.449438 |
| Support Vector Classifier| 0.982759  | 0.974855 | 0.897638 |
| Logistic Regression      | 0.979798  | 0.958414 | 0.818565 |
| Gaussian Naive Bayes     | 0.520737  | 0.875242 | 0.636620 |

### Why Bernoulli Naive Bayes?
- Achieved **highest precision**, meaning almost *no ham messages were falsely classified as spam*  
- High accuracy and F1 score  
- Efficient for binary word occurrence features (presence/absence)

---

## **Techniques Used**
### Data Cleaning  
- Removal of duplicates  
- Label encoding (ham → 0, spam → 1)

### Feature Engineering  
- Created new features:  
  - Number of characters  
  - Number of words  
  - Number of sentences  

### Preprocessing  
- Lowercasing  
- Tokenization  
- Removing punctuation & special characters  
- Removing stop words  
- Stemming  
- TF-IDF vectorization

---

## **How to Run**
1. Install required Python libraries:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn

```



Open the Jupyter Notebook:

jupyter notebook spam_detection.ipynb


Run all cells to perform EDA, preprocessing, training, and evaluation.

Conclusion

Bernoulli Naive Bayes proved to be the most effective model for SMS spam classification due to its perfect precision and competitive accuracy.
Its simplicity, speed, and strong performance make it well-suited for real-world spam filtering applications.

Future Improvements

Potential enhancements include:

Integrating deep learning models (LSTM, BiLSTM, CNN)

Using transformer-based language models (BERT, DistilBERT)

Deploying the model as a Web App using Flask or Streamlit

Implementing real-time message filtering

Exploring better handling for imbalanced data (SMOTE, class weights)
