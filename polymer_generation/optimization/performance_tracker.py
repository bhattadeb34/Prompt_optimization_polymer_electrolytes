from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EpochInfo:
    epoch_number: int
    strategy: str
    prompt: str
    results: Dict
    metrics: Dict
    performance: float
    batch_performances: List[float]
    analysis: Dict = None

class PerformanceTracker:
    """Track optimization performance"""
    def __init__(self, initial_strategy: str):
        self.results = {
            'initial_strategy': initial_strategy,
            'epoch_results': [],
            'best_performance': 0,
            'best_epoch': None,
            'final_summary': {
                'best_strategy': None,
                'best_prompt': None,
                'best_performance': 0,
                'best_molecules': None,
                'performance_trajectory': [],
                'batch_performances': []
            }
        }
    
    def add_epoch_result(self, epoch_info: EpochInfo):
        """Add epoch results"""
        self.results['epoch_results'].append(vars(epoch_info))
        self.results['final_summary']['performance_trajectory'].append(epoch_info.performance)
        self.results['final_summary']['batch_performances'].append(epoch_info.batch_performances)
        
    def update_best(self, epoch_number: int, strategy: str, prompt: str, 
                   performance: float, molecules: Dict):
        """Update best performance"""
        self.results['best_performance'] = performance
        self.results['best_epoch'] = epoch_number
        self.results['final_summary'].update({
            'best_strategy': strategy,
            'best_prompt': prompt,
            'best_performance': performance,
            'best_molecules': molecules
        })