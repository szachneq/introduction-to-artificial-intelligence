import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset

df, _ = load_dataset('wine')

correlation_matrix = df.corr()

high_corr = correlation_matrix[abs(correlation_matrix) > 0.75]

plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.tight_layout(pad=3.0)
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(high_corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.tight_layout(pad=3.0)
plt.show()