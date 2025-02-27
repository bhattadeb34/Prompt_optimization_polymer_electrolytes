import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt

def split_using_kmeans(df: pd.DataFrame, 
                      n_samples: int = 200, 
                      train_frac: float = 0.8, 
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using k-means clustering"""
    n_train = int(n_samples * train_frac)
    n_test = n_samples - n_train
    
    X = df['conductivity'].values.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_samples, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(X)
    
    selected_indices = []
    for i in range(n_samples):
        cluster_points = X[df['cluster'] == i]
        if len(cluster_points) > 0:
            cluster_center = kmeans.cluster_centers_[i]
            closest_idx = df[df['cluster'] == i].index[
                np.abs(df[df['cluster'] == i]['conductivity'].values - cluster_center).argmin()
            ]
            selected_indices.append(closest_idx)
    
    np.random.seed(random_state)
    np.random.shuffle(selected_indices)
    train_indices = selected_indices[:n_train]
    test_indices = selected_indices[n_train:]
    
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    _plot_distributions(df, train_df, test_df)
    _print_split_summary(df, train_df, test_df)
    
    return train_df[['mol_smiles', 'conductivity']], test_df[['mol_smiles', 'conductivity']]

def _plot_distributions(df: pd.DataFrame, 
                       train_df: pd.DataFrame, 
                       test_df: pd.DataFrame) -> None:
    """Plot distribution comparison"""
    plt.figure(figsize=(10, 6))
    plt.hist(df['conductivity'], bins=50, alpha=0.3, 
             label=f'Full Dataset (n={len(df)})', density=True, color='gray')
    plt.hist(train_df['conductivity'], bins=50, alpha=0.5, 
             label=f'Train (n={len(train_df)})', density=True)
    plt.hist(test_df['conductivity'], bins=50, alpha=0.5, 
             label=f'Test (n={len(test_df)})', density=True)
    plt.xlabel('Log Conductivity')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def _print_split_summary(df: pd.DataFrame, 
                        train_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> None:
    """Print summary of data split"""
    print("\nSplit Summary:")
    print(f"Full dataset: {len(df)} molecules")
    print(f"Train set: {len(train_df)} molecules")
    print(f"Test set: {len(test_df)} molecules")
    print(f"\nFull dataset conductivity range: {df['conductivity'].min():.2e} to {df['conductivity'].max():.2e}")
    print(f"Train conductivity range: {train_df['conductivity'].min():.2e} to {train_df['conductivity'].max():.2e}")
    print(f"Test conductivity range: {test_df['conductivity'].min():.2e} to {test_df['conductivity'].max():.2e}")