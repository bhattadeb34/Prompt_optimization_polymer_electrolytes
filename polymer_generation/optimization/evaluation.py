from typing import Dict, List
import numpy as np  # Added this import
from ..generation.polymer_generator import PolymerGenerator
from .strategy_optimizer import StrategyOptimizer
from .performance_tracker import PerformanceTracker, EpochInfo
from ..prompts.evaluation_prompts import EvaluationPromptTemplates

class StrategyEvaluator:
    """Evaluate polymer generation strategy"""
    
    def __init__(self, 
                polymer_generator: PolymerGenerator,
                evaluation_strategy: str):
        """
        Initialize evaluator
        
        Args:
            polymer_generator: Initialized PolymerGenerator
            evaluation_strategy: Best strategy to evaluate
        """
        self.generator = polymer_generator
        self.evaluation_strategy = evaluation_strategy
        self.tracker = PerformanceTracker(evaluation_strategy)
    
    def evaluate(self, 
                batch_data: Dict,
                n_per_molecule: int = 5,
                target_conductivity: float = 1e-3) -> Dict:
        """Run single epoch evaluation"""
        print("\nSTARTING EVALUATION")
        print("="*80)
        print(f"Total batches: {len(batch_data['batches'])}")
        print(f"Molecules per batch: {len(batch_data['batches']['batch_0']['smiles'])}")
        print(f"Generations per molecule: {n_per_molecule}")
        print(f"Evaluation strategy: {self.evaluation_strategy}")
        print("="*80)

        # Process evaluation
        epoch_results, epoch_metrics, epoch_performance, batch_performances = \
            self._process_epoch(batch_data, self.evaluation_strategy, n_per_molecule, target_conductivity)
        
        # Store results
        all_smiles = [
            smiles 
            for batch_info in batch_data['batches'].values()
            for smiles in batch_info['smiles']
        ]
        
        prompt = self._construct_strategy_prompt(
            all_smiles=all_smiles,
            target_conductivity=target_conductivity,
            n_per_molecule=n_per_molecule,
            strategy=self.evaluation_strategy
        )
        
        # Initialize best values
        self.tracker.add_epoch_result(EpochInfo(
            epoch_number=0,
            strategy=self.evaluation_strategy,
            prompt=prompt,
            results=epoch_results,
            metrics=epoch_metrics,
            performance=epoch_performance,
            batch_performances=batch_performances
        ))
        
        self.tracker.update_best(
            epoch_number=0,
            strategy=self.evaluation_strategy,
            prompt=prompt,
            performance=epoch_performance,
            molecules=epoch_results
        )
        
        return self.tracker.results

    def _process_epoch(self, batch_data, strategy, n_per_molecule, target_conductivity):
        """Process single epoch"""
        epoch_results = {}
        epoch_metrics = {}
        batch_performances = []
        
        for batch_id, batch_info in batch_data['batches'].items():
            result, batch_results, batch_metrics = self.generator.generate_batch(
                starting_smiles_batch=batch_info['smiles'],
                target_conductivity=target_conductivity,
                strategy=strategy,
                n_per_molecule=n_per_molecule
            )
            
            if batch_results:
                epoch_results.update(batch_results)
                epoch_metrics.update(batch_metrics)
                batch_perf = StrategyOptimizer._calculate_batch_performance(batch_results)
                batch_performances.append(batch_perf)
        
        epoch_performance = self._calculate_epoch_performance(epoch_results)
        
        return epoch_results, epoch_metrics, epoch_performance, batch_performances

    def _calculate_epoch_performance(self, epoch_results):
        """Calculate comprehensive epoch performance"""
        if not epoch_results:
            return 0
            
        # Collect all improvement factors
        improvements = [
            gen['improvement_factor']
            for gens in epoch_results.values()
            for gen in gens
        ]
        
        # Calculate metrics
        mean_improvement = np.mean(improvements)
        success_rate = np.mean([imp > 1.0 for imp in improvements])
        consistency = 1.0 / (1.0 + np.std(improvements))
        
        # Combine metrics
        performance = (
            0.5 * mean_improvement +
            0.3 * success_rate +
            0.2 * consistency
        )
        
        print(f"\nEpoch Performance Metrics:")
        print(f"Mean Improvement: {mean_improvement:.3f}x")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Consistency: {consistency:.3f}")
        print(f"Combined Performance: {performance:.3f}")
        
        return performance

    def _construct_strategy_prompt(self, all_smiles, target_conductivity, n_per_molecule, strategy):
        """
        Construct prompt using EvaluationPromptTemplates
        
        Args:
            all_smiles: List of SMILES strings
            target_conductivity: Target conductivity value
            n_per_molecule: Number of generations per molecule
            strategy: Best strategy to use
        """
        molecule_pairs = "\n".join(
            f"{i+1}. SMILES: {smiles}, Target Conductivity: {target_conductivity:.2e} mS/cm"
            for i, smiles in enumerate(all_smiles)
        )
        
        return EvaluationPromptTemplates.construct_evaluation_prompt(
            molecule_pairs=molecule_pairs,
            n_per_molecule=n_per_molecule,
            best_strategy=strategy
        )