import numpy as np
import pandas as pd
import seaborn
import sklearn
from sklearn.datasets import fetch_california_housing


def to_df(dataset):
    return pd.DataFrame(
        data=np.c_[dataset["data"], dataset["target"]],
        columns=dataset["feature_names"] + ["target"],
    )


def load_dataset(dataset_name: str):
    if dataset_name == "california_housing":
        dataset = fetch_california_housing()
        dataset = to_df(dataset)
    elif dataset_name == "diabetes":
        dataset = sklearn.datasets.load_diabetes()
        dataset = to_df(dataset)
    elif dataset_name == "iris":
        dataset = sklearn.datasets.load_iris()
        dataset = to_df(dataset)
        dataset["target"] = dataset["target"].astype(int)
    elif dataset_name == "titanic":
        dataset = seaborn.load_dataset("titanic")
        # Replace 'survived' column with 'target' for consistency
        dataset["target"] = dataset["survived"]
        dataset = dataset.drop(columns=["survived"])
        dataset["target"] = dataset["target"].astype(int)
    elif dataset_name == "wine":
        dataset = sklearn.datasets.load_wine()
        dataset = to_df(dataset)
        dataset["target"] = dataset["target"].astype(int)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    return X, y