from typing import List, Dict, Tuple
import random
import numpy as np

class BatchProcessor:
    @staticmethod
    def create_batches(smiles_list: List[str], 
                    target_conductivity: float,
                    batch_size: int) -> Dict:
        """
        Create batches from SMILES list with target conductivity
        
        Args:
            smiles_list: List of SMILES strings
            target_conductivity: Target conductivity value
            batch_size: Size of each batch
            
        Returns:
            Dictionary containing batch information:
                - batches: Dictionary of batches with their info
                - batch_size: Size of each batch
                - total_molecules: Total number of molecules
        """
        # Calculate number of batches
        n_batches = (len(smiles_list) + batch_size - 1) // batch_size
        
        # Create batches dictionary
        batches = {
            f'batch_{i}': {
                'smiles': smiles_list[i:i + batch_size],
                'target_conductivity': target_conductivity
            }
            for i in range(n_batches)
        }
        
        return {
            'batches': batches,
            'batch_size': batch_size,
            'total_molecules': len(smiles_list)
        }

    @staticmethod
    def calculate_batch_performance(batch_results: Dict) -> float:
        """Calculate average improvement factor for a batch"""
        if not batch_results:
            return 0
        
        improvements = []
        for generations in batch_results.values():
            if generations:
                best_gen = max(generations, key=lambda x: x['improvement_factor'])
                improvements.append(best_gen['improvement_factor'])
        
        return np.mean(improvements) if improvements else 0
    
    @staticmethod
    def split_train_test(X: List, y: List, 
                        test_size: float = 0.2, 
                        random_seed: int = 42) -> Tuple[List, List, List, List]:
        """Split data into train and test sets"""
        random.seed(random_seed)
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        split = int(len(X) * (1 - test_size))
        
        train_X = [X[i] for i in indices[:split]]
        train_y = [y[i] for i in indices[:split]]
        test_X = [X[i] for i in indices[split:]]
        test_y = [y[i] for i in indices[split:]]
        
        return train_X, test_X, train_y, test_y