import json
import glob
import os
from datetime import datetime
from typing import Dict
import numpy as np
def save_optimization_results(optimization_results, base_dir="results"):
    """
    Save optimization results to JSON with timestamp
    
    Args:
        optimization_results: Results from optimize_strategy
        base_dir: Directory to save results
    """
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create results directory with timestamp
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir)
    
    # Prepare results for JSON serialization (convert numpy types to Python types)
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Save full results
    results_file = os.path.join(results_dir, "optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2, default=convert_for_json)
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'n_epochs': len(optimization_results['epoch_results']),
        'best_epoch': optimization_results['best_epoch'],
        'best_performance': float(optimization_results['best_performance']),
        'performance_trajectory': [float(p) for p in optimization_results['final_summary']['performance_trajectory']]
    }
    
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved in: {results_dir}")
    print(f"Full results: {results_file}")
    print(f"Summary: {summary_file}")
    
    return results_dir


def load_latest_results(prefix: str = "opt_run") -> Dict:
    """Load most recent results file"""
    files = glob.glob(f"{prefix}_*.json")
    
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}'")
    
    latest_file = max(files, key=os.path.getmtime)
    
    print(f"Loading latest results from '{latest_file}'")
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results

def load_optimization_results(results_dir):
    """Load optimization results from directory"""
    results_file = os.path.join(results_dir, "optimization_results.json")
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results