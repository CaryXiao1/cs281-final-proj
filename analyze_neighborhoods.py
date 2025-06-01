import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Analyze neighborhood structure of admission decisions')
    parser.add_argument('model', choices=['chatgpt', 'deepseek'], help='Choose between ChatGPT or DeepSeek model')
    args = parser.parse_args()
    
    # Select input file based on model
    input_file = 'out.csv' if args.model == 'chatgpt' else 'out-deepseek.csv'
    
    # Read the CSV file
    df = pd.read_csv(input_file, header=None, names=['Race', 'Gender', 'GPA', 'ACT', 'Income', 'Location', 'School', 'Rank', 'Decision'])

    # convert categorical variables to numeric using one-hot encoding
    categorical_cols = ['Race', 'Gender', 'ACT', 'Income', 'Location', 'School', 'Rank']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    df_encoded = df_encoded.iloc[1:]

    # Multiply race and gender columns by 4
    race_cols = [col for col in df_encoded.columns if 'Race_' in col]
    gender_cols = [col for col in df_encoded.columns if 'Gender_' in col]

    # Convert bools to ints and multiply by 4
    # df_encoded[race_cols] = df_encoded[race_cols].astype(int) * 1.5
    # df_encoded[gender_cols] = df_encoded[gender_cols].astype(int) * 1.5

    # Create binary decision column (1 for ADMIT, 0 for REJECT)
    df_encoded['Decision'] = (df_encoded['Decision'] == 'ADMIT').astype(int)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=25)
    projected_data = tsne.fit_transform(df_encoded.drop('Decision', axis=1))

    # Find k-nearest neighbors
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(projected_data)  # k+1 because the point itself is included
    distances, indices = nbrs.kneighbors(projected_data)

    # Calculate proportion of neighbors with same label for each point
    proportions = []
    for i in range(len(projected_data)):
        # Get the labels of the k nearest neighbors (excluding the point itself)
        neighbor_labels = df_encoded['Decision'].iloc[indices[i][1:]]  # Skip first index as it's the point itself
        # Calculate proportion of neighbors with same label
        same_label_prop = np.mean(neighbor_labels == df_encoded['Decision'].iloc[i])
        proportions.append(same_label_prop)

    # Calculate average proportion across all points
    avg_proportion = np.mean(proportions)
    print(f"Average proportion of neighbors with same label: {avg_proportion:.3f}")

    # Create plot with color intensity based on neighborhood homogeneity
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(projected_data[:, 0], 
                         projected_data[:, 1], 
                         c=proportions,
                         cmap='viridis',
                         alpha=0.6)
    plt.colorbar(scatter, label='Proportion of neighbors with same label')
    plt.title(f't-SNE Projection with Neighborhood Homogeneity ({args.model.upper()})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

if __name__ == "__main__":
    main() 