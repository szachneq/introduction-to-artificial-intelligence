import random
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from datasets import load_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # Load dataset
    X, y = load_dataset("wine")
    # Preprocess dataset
    
    X = X.drop(columns=["color_intensity"])
    X = X.drop(columns=["proline"])
    X = X.drop(columns=["flavanoids"])
    
    # X = X.drop(columns=["ash"])
    # X = X.drop(columns=["nonflavanoid_phenols"])
    # X = X.drop(columns=["magnesium"])
    # X = X.drop(columns=["alcalinity_of_ash"])
    # X = X.drop(columns=["malic_acid"])
    # X = X.drop(columns=["proanthocyanins"])

# Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the models
    # model1 = LogisticRegression(max_iter=1000, random_state=seed)
    model1 = SVC(random_state=seed)  # Use SVM for model1
    model2 = RandomForestClassifier(random_state=seed)

    # Evaluate models using cross-validation on scaled data
    scoring = make_scorer(accuracy_score)
    # scoring = make_scorer(precision_score, average='micro')
    # scoring = make_scorer(recall_score, average='micro')
    # scoring = make_scorer(f1_score, average='micro')
    scores1 = cross_val_score(model1, X_train_scaled, y_train, cv=4, scoring=scoring)
    scores2 = cross_val_score(model2, X_train_scaled, y_train, cv=4, scoring=scoring)

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


if __name__ == "__main__":
    X, y = load_dataset("wine")
    main()
