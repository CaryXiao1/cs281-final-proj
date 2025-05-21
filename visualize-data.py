import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('out.csv', header=None, names=['Race', 'Gender', 'GPA', 'ACT', 'Income', 'Location', 'School', 'Rank', 'Decision'])

# convert categorical variables to numeric using one-hot encoding
categorical_cols = ['Race', 'Gender', 'ACT', 'Income', 'Location', 'School', 'Rank']
df_encoded = pd.get_dummies(df, columns=categorical_cols)
df_encoded = df_encoded.iloc[1:]
print(df_encoded)

# Create binary decision column (1 for ADMIT, 0 for REJECT)
df_encoded['Decision'] = (df_encoded['Decision'] == 'ADMIT').astype(int)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=25)
projected_data = tsne.fit_transform(df_encoded.drop('Decision', axis=1))

# create plot
plt.figure(figsize=(10, 8))
plt.scatter(projected_data[df_encoded['Decision'] == 1, 0], 
           projected_data[df_encoded['Decision'] == 1, 1], 
           c='green', label='ADMIT')
plt.scatter(projected_data[df_encoded['Decision'] == 0, 0], 
           projected_data[df_encoded['Decision'] == 0, 1], 
           c='red', label='REJECT')
plt.title('t-SNE Projection of College Admissions Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()
