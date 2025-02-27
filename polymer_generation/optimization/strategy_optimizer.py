from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from .performance_tracker import PerformanceTracker, EpochInfo
from ..generation.polymer_generator import PolymerGenerator
from ..prompts.prompt_templates import PromptTemplates
from ..schemas import StrategyResponse

class StrategyOptimizer:
    """Optimize polymer generation strategy"""
    def __init__(self, 
                polymer_generator: PolymerGenerator,
                initial_strategy: str):
        self.generator = polymer_generator
        self.initial_strategy = initial_strategy
        self.tracker = PerformanceTracker(initial_strategy)
    
    def get_worst_performing_examples(self, validated_results: dict, 
                                    n_examples: int = 3) -> dict:
        """Get worst performing examples"""
        best_results = self.get_best_generations(validated_results)
        parent_improvements = []
        for parent, gens in best_results.items():
            gen = gens[0]
            improvement_factor = gen['actual_conductivity'] / gen['parent_conductivity']
            parent_improvements.append((parent, improvement_factor))
        worst_parents = sorted(parent_improvements, key=lambda x: x[1])[:n_examples]
        return {parent: best_results[parent] for parent, _ in worst_parents}
    
    def _construct_strategy_prompt(self, 
                                 all_smiles: List[str], 
                                 batch_data: Dict, 
                                 n_per_molecule: int, 
                                 strategy: str) -> str:
        """Construct prompt for strategy generation"""
        return PromptTemplates.construct_generation_prompt(
            molecule_pairs="\n".join(
                f"{i+1}. SMILES: {smiles}, Target Conductivity: "
                f"{batch_data['batches']['batch_0']['target_conductivity']:.2e} mS/cm"
                for i, smiles in enumerate(all_smiles)
            ),
            n_per_molecule=n_per_molecule,
            strategy=strategy
        )
    def _generate_new_strategy(self, epoch_results: Dict, current_strategy: str) -> Tuple[str, Dict]:
        """Generate new strategy based on results"""
        print("\nGENERATING NEW STRATEGY")
        print("-"*50)
        print("Current strategy:", current_strategy)
        
        print("\nPreparing performance summary...")
        # Prepare history summary
        history = []
        total_improvement = 0
        total_pairs = 0
        
        # Get best generations for each parent
        print("\nBest generations from current epoch:")
        best_results = {}
        for parent, gens in epoch_results.items():
            if gens:
                best_gen = max(gens, key=lambda x: x['improvement_factor'])
                print(f"\nParent: {parent}")
                print(f"Best generation: {best_gen['generated_smiles']}")
                print(f"Improvement: {best_gen['improvement_factor']:.2f}x")
                print(f"Explanation: {best_gen['explanation']}")
                best_results[parent] = [best_gen]
                
                # Format history entry more generally
                history.append(
                    f"Parent: {parent}\n"
                    f"Best Generation: {best_gen['generated_smiles']}\n"
                    f"Improvement Factor: {best_gen['improvement_factor']:.2f}x\n"
                    f"Explanation: {best_gen['explanation']}\n"
                )
                total_improvement += best_gen['improvement_factor']
                total_pairs += 1
        
        history_str = "\n".join(history)
        avg_improvement = total_improvement / total_pairs if total_pairs > 0 else 0
        print(f"\nAverage improvement: {avg_improvement:.2f}x")
        
        print("\nGenerating new strategy...")
        # Get new strategy from LLM
        result = self.generator._get_llm_response(
            PromptTemplates.get_strategy_prompt(
                previous_strategy=current_strategy,
                history_str=history_str,
                avg_improvement=avg_improvement
            ),
            schema=StrategyResponse  # Make sure this matches schemas.py
        )
        print("\nNew strategy generated:")
        print(result.strategy)
        print("\nAnalysis:")
        print("Strengths:", result.analysis['strengths'])
        print("Weaknesses:", result.analysis['weaknesses'])
        print("Recommendations:", result.analysis['recommendations'])
        return result.strategy, result.analysis
    
    def process_initial_strategy(self, batch_data: Dict, n_per_molecule: int) -> Tuple:
        """Process initial strategy"""
        initial_results = {}
        initial_batch_performances = []
        initial_metrics = {}
        
        for batch_id, batch_info in batch_data['batches'].items():
            try:
                result, batch_results, batch_metrics = self.generator.generate_batch(
                    starting_smiles_batch=batch_info['smiles'],
                    target_conductivity=batch_info['target_conductivity'],
                    strategy=self.initial_strategy,
                    n_per_molecule=n_per_molecule
                )
                
                if batch_results:
                    initial_results.update(batch_results)
                    initial_metrics.update(batch_metrics)
                    batch_perf = self._calculate_batch_performance(batch_results)
                    initial_batch_performances.append(batch_perf)
                    
            except Exception as e:
                print(f"Error processing initial batch {batch_id}: {str(e)}")
                continue
                
        return initial_results, initial_batch_performances, initial_metrics
    def optimize(self, 
                batch_data: Dict,
                n_epochs: int = 10,
                n_per_molecule: int = 5) -> Dict:
        """Run optimization"""
        print("\n" + "="*80)
        print("STARTING OPTIMIZATION PROCESS")
        print("="*80)
        print(f"Total batches: {len(batch_data['batches'])}")
        print(f"Molecules per batch: {batch_data['batch_size']}")
        print(f"Generations per molecule: {n_per_molecule}")
        print(f"Number of epochs: {n_epochs}")
        print(f"Initial strategy: {self.initial_strategy}")
        print("="*80)

        print("\nPROCESSING INITIAL STRATEGY")
        print("-"*50)
        # Process initial strategy
        initial_results, initial_batch_performances, initial_metrics = \
            self.process_initial_strategy(batch_data, n_per_molecule)
        
        # Setup initial state
        all_smiles = [smiles for batch_info in batch_data['batches'].values() 
                    for smiles in batch_info['smiles']]
        current_strategy = self.initial_strategy
        
        # Calculate initial performance
        initial_performance = self._calculate_epoch_performance(initial_results)  # Use new performance calculation
        
        # Store initial results
        initial_prompt = self._construct_strategy_prompt(
            all_smiles, batch_data, n_per_molecule, current_strategy
        )
        
        # Initialize best values with initial results
        best_performance = initial_performance
        best_strategy = current_strategy
        best_epoch = 0
        best_results = initial_results
        
        self.tracker.add_epoch_result(EpochInfo(
            epoch_number=0,
            strategy=current_strategy,
            prompt=initial_prompt,
            results=initial_results,
            metrics=initial_metrics,
            performance=initial_performance,
            batch_performances=initial_batch_performances
        ))
        
        # Update tracker with initial best
        self.tracker.update_best(
            epoch_number=0,
            strategy=current_strategy,
            prompt=initial_prompt,
            performance=initial_performance,
            molecules=initial_results
        )
        
   
        # Optimization loop
        for epoch in tqdm(range(n_epochs), desc="Optimizing Strategy"):
            # Process epoch
            epoch_results, epoch_metrics, epoch_performance, batch_performances = \
                self._process_epoch(batch_data, current_strategy, n_per_molecule)
            
            new_strategy, analysis = self._generate_new_strategy(
                epoch_results, current_strategy
            )
            
            # Store epoch information
            current_prompt = self._construct_strategy_prompt(
                all_smiles, batch_data, n_per_molecule, new_strategy
            )
            
            self.tracker.add_epoch_result(EpochInfo(
                epoch_number=epoch + 1,
                strategy=new_strategy,
                prompt=current_prompt,
                results=epoch_results,
                metrics=epoch_metrics,
                performance=epoch_performance,
                batch_performances=batch_performances,
                analysis=analysis
            ))
             
            # Update if better using new metric
            if epoch_performance > best_performance:
                print(f"\nImproved Performance: {epoch_performance:.3f} > {best_performance:.3f}")
                best_performance = epoch_performance
                best_strategy = new_strategy
                current_strategy = new_strategy
                
                self.tracker.update_best(
                    epoch_number=epoch + 1,
                    strategy=new_strategy,
                    prompt=current_prompt,
                    performance=epoch_performance,
                    molecules=epoch_results
                )
            else:
                print(f"\nNo improvement: {epoch_performance:.3f} <= {best_performance:.3f}")
                current_strategy = best_strategy

        return self.tracker.results

    def _calculate_epoch_performance(self, epoch_results):
        """Calculate comprehensive epoch performance"""
        if not epoch_results:
            return 0
            
        # Collect all improvement factors
        improvements = [
            gen['improvement_factor']
            for gens in epoch_results.values()  # epoch_results is dict of parent -> generations
            for gen in gens                     # gens is list of generation dicts
        ]
        
        # Calculate metrics
        mean_improvement = np.mean(improvements)
        success_rate = np.mean([imp > 1.0 for imp in improvements])
        consistency = 1.0 / (1.0 + np.std(improvements))  # Normalized to [0,1]
        
        # Combine metrics
        performance = (
            0.5 * mean_improvement +  # Emphasize average improvement
            0.3 * success_rate +      # Reward consistent success
            0.2 * consistency        # Reward stability
        )
        
        print(f"\nEpoch Performance Metrics:")
        print(f"Mean Improvement: {mean_improvement:.3f}x")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Consistency: {consistency:.3f}")
        print(f"Combined Performance: {performance:.3f}")
        
        return performance


    @staticmethod
    def _calculate_batch_performance(batch_results):
        """Calculate batch performance from list of generations"""
        if not batch_results:
            return 0
        
        improvements = [
            gen['improvement_factor']
            for gens in batch_results.values()
            for gen in gens
        ]
        return np.mean(improvements)

    def _process_epoch(self, batch_data, strategy, n_per_molecule):
        """Process single epoch"""
        epoch_results = {}
        epoch_metrics = {}
        batch_performances = []
        
        for batch_id, batch_info in batch_data['batches'].items():
            result, batch_results, batch_metrics = self.generator.generate_batch(
                starting_smiles_batch=batch_info['smiles'],
                target_conductivity=batch_info['target_conductivity'],
                strategy=strategy,
                n_per_molecule=n_per_molecule
            )
            
            if batch_results:
                epoch_results.update(batch_results)
                epoch_metrics.update(batch_metrics)
                batch_perf = StrategyOptimizer._calculate_batch_performance(batch_results)  # Call static method
                batch_performances.append(batch_perf)
        
        # Calculate overall epoch performance
        epoch_performance = self._calculate_epoch_performance(epoch_results)
        
        return epoch_results, epoch_metrics, epoch_performance, batch_performances