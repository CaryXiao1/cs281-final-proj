import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Visualize college admissions data')
    parser.add_argument('model', choices=['chatgpt', 'deepseek', 'gemini'], help='Choose between ChatGPT, DeepSeek, or Gemini model')
    args = parser.parse_args()
    
    # Select input file based on model
    input_file = {
        'chatgpt': 'out.csv',
        'deepseek': 'out-deepseek.csv',
        'gemini': 'out-gemini.csv'
    }[args.model]
    
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
    plt.title(f't-SNE Projection of College Admissions Data ({args.model.upper()})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
