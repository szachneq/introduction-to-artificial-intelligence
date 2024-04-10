import random

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from datasets import load_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # TODO Load and preprocess dataset
    X, y = load_dataset(...)
    ...

    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # TODO Define the models
    model1 = ...
    model2 = ...

    # TODO evaluate model using cross-validation
    scores1 = cross_val_score(model1, X_train, y_train, cv=4, scoring=...)
    scores2 = cross_val_score(model2, X_train, y_train, cv=4, scoring=...)

    # Fit the best model on the entire training set and get the predictions
    final_model1 = model1.fit(X_train, y_train)
    final_model2 = model2.fit(X_train, y_train)

    predictions1 = final_model1.predict(X_test)
    predictions2 = final_model2.predict(X_test)

    # TODO Evaluate the final predictions with the metric of your choice
    ...