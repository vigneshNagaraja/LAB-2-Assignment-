# LAB-2-Assignment
Chronic Kidney Disease Classification Using Machine Learning

#Step 1: Upload and Load Data

from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv('ckd_cleaned_imputed.csv')
df.head()

 #Step 2: Basic Cleaning (Whitespace + Categorical Encoding)

# Clean up any stray tabs or spaces
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Encode categories and target
binary_map = {
    'yes': 1, 'no': 0,
    'present': 1, 'notpresent': 0,
    'abnormal': 1, 'normal': 0,
    'good': 1, 'poor': 0,
    'ckd': 1, 'notckd': 0
}
df.replace(binary_map, inplace=True)

 Evaluation Metrics
 Accuracy
 Precision
 Recall
 F1-Score
 Confusion Matrix


 Dataset
UCI Machine Learning Repository
ckd_cleaned_imputed.csv

 Outcome
All three models performed well.the Decision Tree achieved the highest F1-score (0.99) and offered strong interpretability for clinical use.
