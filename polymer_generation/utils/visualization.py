import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np

def analyze_optimization_results(optimization_results):
    """Analysis focusing on parent vs generated molecules conductivity"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Collect data from all epochs with proper parent indexing
    all_data = []
    parent_molecules = set()
    
    # First collect all unique parent molecules across all epochs
    for epoch in optimization_results['epoch_results']:
        parent_molecules.update(epoch['results'].keys())
    
    # Create parent dictionary
    parent_dict = {smiles: idx for idx, smiles in enumerate(sorted(parent_molecules))}
    
    # Collect all data
    for epoch in optimization_results['epoch_results']:
        for parent_smiles, generations in epoch['results'].items():
            parent_idx = parent_dict[parent_smiles]
            for gen in generations:
                all_data.append({
                    'parent_smiles': parent_smiles,
                    'parent_idx': parent_idx,
                    'conductivity': gen['actual_conductivity'],
                    'parent_conductivity': gen['parent_conductivity'],
                    'target_conductivity': gen['target_conductivity'],
                    'improvement_factor': gen['improvement_factor']
                })
    
    df = pd.DataFrame(all_data)
    
    # 1. Distribution of Conductivities
    violin_data = pd.DataFrame({
        'Conductivity': list(df['parent_conductivity'].unique()) + list(df['conductivity']),
        'Type': ['Parent']*len(df['parent_conductivity'].unique()) + ['Generated']*len(df['conductivity'])
    })
    
    sns.violinplot(data=violin_data, x='Type', y='Conductivity', ax=ax1)
    ax1.set_title('Distribution of Parent vs Generated Conductivities')
    ax1.set_ylabel('Conductivity (mS/cm)')
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of Improvement Ratios
    sns.histplot(data=df['improvement_factor'], ax=ax2, bins=50, stat='density', kde=True)
    ax2.axvline(1.0, color='r', linestyle='--', label='No Improvement')
    ax2.set_title('Distribution of Improvement Factors')
    ax2.set_xlabel('Improvement Factor')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Parent vs Generated Conductivity
    scatter = ax3.scatter(df['parent_idx'], 
                         df['conductivity'],
                         c=df['conductivity'],
                         cmap='viridis',
                         alpha=0.6)
    
    # Add parent conductivities
    parent_data = df.groupby('parent_idx').first()
    ax3.scatter(parent_data.index, 
                parent_data['parent_conductivity'],
                color='red',
                marker='*',
                s=100,
                label='Parent')
    
    ax3.set_title('Parent vs Generated Conductivity')
    ax3.set_xlabel('Parent Index')
    ax3.set_ylabel('Conductivity (mS/cm)')
    plt.colorbar(scatter, ax=ax3, label='Generated Conductivity (mS/cm)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Optional: Add some parent SMILES as x-tick labels (maybe every nth parent)
    n = max(1, len(parent_molecules) // 10)  # Show about 10 labels
    selected_indices = range(0, len(parent_molecules), n)
    ax3.set_xticks(selected_indices)
    ax3.set_xticklabels([f"P{i}" for i in selected_indices], rotation=45)

    # 4. Performance Trajectory
    performances = [epoch['performance'] for epoch in optimization_results['epoch_results']]
    epochs = range(len(performances))
    
    ax4.plot(epochs, performances, 'b-o', label='Per Epoch Performance', alpha=0.5)
    ax4.plot(epochs, np.maximum.accumulate(performances), 'r-o', label='Best So Far')
    ax4.set_title('Performance Trajectory')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Performance')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of unique parent molecules: {len(parent_molecules)}")
    print(f"Total generated molecules: {len(df)}")
    print("\nImprovement Factors:")
    print(f"Mean: {df['improvement_factor'].mean():.2f}x")
    print(f"Max: {df['improvement_factor'].max():.2f}x")
    print(f"Percentage of improvements: {(df['improvement_factor'] > 1).mean()*100:.1f}%")
    
    print("\nConductivity Statistics:")
    print("Parent Conductivities:")
    print(f"  Mean: {df['parent_conductivity'].mean():.4f}")
    print(f"  Min: {df['parent_conductivity'].min():.4f}")
    print(f"  Max: {df['parent_conductivity'].max():.4f}")
    print("\nGenerated Conductivities:")
    print(f"  Mean: {df['conductivity'].mean():.4f}")
    print(f"  Min: {df['conductivity'].min():.4f}")
    print(f"  Max: {df['conductivity'].max():.4f}")