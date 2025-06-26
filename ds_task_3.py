# STEP 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# STEP 2: Load dataset (✔️ Corrected path & delimiter)
dataset_path = r'C:\Users\ADMIN\Desktop\ds_task_03\bank.csv'  # ✅ Adjust path if needed

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at: {dataset_path}")
    print("💡 Download it from:\nhttps://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset")
else:
    # STEP 3: Load the dataset (FIXED delimiter)
    df = pd.read_csv(dataset_path, sep=',')  # ✅ Use correct separator
    print("✅ Dataset loaded successfully.")
    print("🔎 Columns in dataset:", df.columns.tolist())

    # STEP 4: Check if 'deposit' column exists (target variable)
    if 'deposit' not in df.columns:
        raise ValueError("❌ Column 'deposit' not found. Please check the dataset.")

    # STEP 5: Preprocess data
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # STEP 6: Split the data
    X = df.drop('deposit', axis=1)
    y = df['deposit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # STEP 7: Train the Decision Tree model
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # STEP 8: Make predictions
    y_pred = model.predict(X_test)

    # STEP 9: Evaluate the model
    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    # STEP 10: Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
 