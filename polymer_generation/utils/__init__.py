# __init__.py
try:
    from .api_utils import load_api_key
    from .smiles_utils import SMILESProcessor
    from .batch_utils import BatchProcessor
    from .data_processing import split_using_kmeans
    from .io_utils import save_optimization_results, load_optimization_results
    from .visualization import analyze_optimization_results

    __all__ = [
        'load_api_key',
        'SMILESProcessor',
        'BatchProcessor',
        'split_using_kmeans',
        'save_optimization_results',
        'load_optimization_results',
        'analyze_optimization_results'
    ]
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")