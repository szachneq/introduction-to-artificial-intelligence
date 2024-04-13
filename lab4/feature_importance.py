import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset

seed = 0
random.seed(seed)
np.random.seed(seed)

X, y = load_dataset('wine')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the model
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# Fit the model
clf.fit(X_train, y_train)

# Get feature importances
importances = clf.feature_importances_

# Convert the importances into a DataFrame for easier viewing
feature_importances_df = pd.DataFrame({'feature': X.columns, 'importance': importances})

# Sort the DataFrame to see the most important features at the top
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False).reset_index(drop=True)

# Display the feature importances
print(feature_importances_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout(pad=3.0)
plt.show()
