import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def to_df(dataset):
    return pd.DataFrame(
        data=np.c_[dataset["data"], dataset["target"]],
        columns=dataset["feature_names"] + ["target"],
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 123
    set_seed(seed)

    # Load and preprocess dataset
    dataset = load_wine()
    dataset = to_df(dataset)
    dataset["target"] = dataset["target"].astype(int)
    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    
    # Preprocess dataset
    X = X.drop(columns=["total_phenols"])

    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Normalize the data
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the models
    # model1 = LogisticRegression(max_iter=1000, random_state=seed)
    model1 = SVC(random_state=seed)  # Use SVM for model1
    model2 = RandomForestClassifier(random_state=seed)

    # Evaluate models using cross-validation on scaled data
    scores1 = cross_val_score(model1, X_train_scaled, y_train, cv=4, scoring='accuracy')
    scores2 = cross_val_score(model2, X_train_scaled, y_train, cv=4, scoring='accuracy')

    # print(f"Logistic Regression Cross-Validation Accuracy: {np.mean(scores1):.3f} (+/- {np.std(scores1):.3f})")
    print(f"SVM Cross-Validation Accuracy: {np.mean(scores1):.3f} (+/- {np.std(scores1):.3f})")
    print(f"Random Forest Classifier Cross-Validation Accuracy: {np.mean(scores2):.3f} (+/- {np.std(scores2):.3f})")

    # Fit the best model on the entire training set and get the predictions on scaled data
    final_model1 = model1.fit(X_train_scaled, y_train)
    final_model2 = model2.fit(X_train_scaled, y_train)

    predictions1 = final_model1.predict(X_test_scaled)
    predictions2 = final_model2.predict(X_test_scaled)

    # Evaluate the final predictions with the metric of your choice
    accuracy1 = accuracy_score(y_test, predictions1)
    accuracy2 = accuracy_score(y_test, predictions2)

    # print(f"Logistic Regression Test Accuracy: {accuracy1:.3f}")
    print(f"SVM Test Accuracy: {accuracy1:.3f}")
    print(f"Random Forest Test Accuracy: {accuracy2:.3f}")
