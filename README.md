# LAB-2-Assignment-
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


 #Step 3: Exploratory Data Analysis (Histograms)

important_cols = ['age', 'bp', 'hemo', 'sc', 'class']

for col in important_cols:
    plt.figure(figsize=(6, 4))
    df[col].hist(bins=20, color='teal', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()


#Step 4: Prepare Data for Modeling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features for LR and k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Step 5: Train Models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize models
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=42)

# Train models
lr.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)  # Tree does not need scaling


 #Step 6: Predict and Evaluate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_true, y_pred))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
    plt.title(f"{name} Confusion Matrix")
    plt.grid(False)
    plt.show()

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("k-NN", y_test, y_pred_knn)
evaluate_model("Decision Tree", y_test, y_pred_dt)


 #Step 7: Compare All Models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_scores(y_true, y_pred, model):
    return {
        "Model": model,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

results = [
    get_scores(y_test, y_pred_lr, "Logistic Regression"),
    get_scores(y_test, y_pred_knn, "k-NN"),
    get_scores(y_test, y_pred_dt, "Decision Tree")
]

results_df = pd.DataFrame(results)
results_df
