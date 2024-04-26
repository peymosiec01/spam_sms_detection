# SMS Spam Detection

This repository contains a solution for classifying SMS messages as spam or legitimate using machine learning techniques. The model is trained on the SMS Spam Collection dataset, which consists of tagged SMS messages collected for spam research.

## Dataset

The SMS Spam Collection dataset can be found on [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset). It contains 5,574 SMS messages tagged as either "ham" (legitimate) or "spam".

## Approach

### Data Preprocessing and Feature Engineering
- Text preprocessing was performed using spaCy, including tokenization, lemmatization, and removal of stopwords.
- Feature engineering included calculating the length of each message.

### Model Training
- The dataset was split into training and test sets.
- Various classifiers were trained, including Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).
- Imbalanced class distribution was addressed using Synthetic Minority Over-sampling Technique (SMOTE).
- Hyperparameter tuning was performed using Grid Search and Random Search.

## Results

| Model          | Accuracy | Precision | Recall | F1 Score | AUC   |
|----------------|----------|-----------|--------|----------|-------|
| Naive Bayes   | 0.96     | 0.92      | 0.85   | 0.88     | 0.98  |
| Logistic Regression | 0.97 | 0.93      | 0.88   | 0.90     | 0.97  |
| Linear SVM     | 0.98     | 0.94      | 0.89   | 0.92     | 0.98  |

The Linear SVM model demonstrated the best performance with an accuracy of 0.98 and an AUC of 0.98.

## Instructions

1. Clone the repository.
2. Install dependencies.
3. Run the Jupyter notebook to train and evaluate the models.

## Files

- `spam.csv`: Dataset containing SMS messages.
- `sms_spam_detection.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- `README.md`: This file.

## Acknowledgements

- The SMS Spam Collection dataset was obtained from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset).
- Libraries used: NumPy, pandas, scikit-learn, spaCy, seaborn, Matplotlib, imbalanced-learn.

---

Feel free to customize the README according to your preferences and add any additional information you think would be helpful for users.
