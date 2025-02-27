from typing import Dict, List, Tuple, Any
import numpy as np
import langfun as lf 
from rdkit import Chem
from ..models.llm_config import LLMConfig
from ..utils.smiles_utils import SMILESProcessor
from ..models.polymer_metrics import PolymerMetrics  
from ..model import predict  
from ..prompts.prompt_templates import PromptTemplates  # Added this import
from ..schemas import StrategyResponse, MoleculeGeneration

class PolymerGenerator:
    """Main class for polymer generation"""
    def __init__(self, 
                api_keys: Dict[str, str],
                base_path: str = "/noether/s0/dxb5775/htp_md/htpmd/ml_models",
                model_name: str = "claude-3-5-sonnet",
                temperature: float = 0.0):
        self.api_keys = api_keys
        self.base_path = base_path
        self.model_name = model_name
        self.temperature = temperature
        self.metrics_calculator = PolymerMetrics()
        self.smiles_processor = SMILESProcessor()
    
    def _construct_prompt(self, 
                         molecule_pairs: str, 
                         n_per_molecule: int, 
                         strategy: str,
                         feedback: str = None) -> str:
        """Construct prompt using PromptTemplates"""
        return PromptTemplates.construct_generation_prompt(
            molecule_pairs=molecule_pairs,
            n_per_molecule=n_per_molecule,
            strategy=strategy,
            feedback=feedback
        )
        
    def process_single_generation(self, 
                                gen_smiles: str, 
                                explanation: str, 
                                parent_conductivity: float,
                                target_conductivity: float) -> Dict:
        """Process a single generated SMILES"""
        if not self.smiles_processor.validate_polymer_smiles(gen_smiles)[0]:
            return None
            
        try:
            canonical_gen = Chem.CanonSmiles(gen_smiles)
            preds = predict([canonical_gen], 'conductivity', self.base_path)
            
            if len(preds) > 0:
                conductivity = float(preds[0]) * 1000  # Convert to mS/cm
                improvement_factor = conductivity / parent_conductivity if parent_conductivity > 0 else 0
                
                return {
                    'generated_smiles': canonical_gen,
                    'actual_conductivity': conductivity,
                    'parent_conductivity': parent_conductivity,
                    'improvement_factor': improvement_factor,
                    'target_conductivity': target_conductivity,
                    'error': abs(conductivity - target_conductivity) / target_conductivity,
                    'explanation': explanation
                }
        except Exception as e:
            print(f"Error processing generation {gen_smiles}: {str(e)}")
        
        return None
        

    def generate_batch(self,
                      starting_smiles_batch: List[str],
                      target_conductivity: float,
                      strategy: str,
                      n_per_molecule: int = 5) -> Tuple[Any, Dict, Dict]:
        
        """Generate polymers for a batch of molecules"""
        print("\nPROCESSING BATCH")
        print("-"*50)
        print(f"Number of molecules in batch: {len(starting_smiles_batch)}")
        print(f"Target conductivity: {target_conductivity:.2e} mS/cm")
        print(f"Generations per molecule: {n_per_molecule}")
        # Prepare input
        print("\nPreparing LLM prompt...")
        molecule_pairs = "\n".join(
            f"{i+1}. SMILES: {smiles}, Target Conductivity: {target_conductivity:.2e} mS/cm"
            for i, smiles in enumerate(starting_smiles_batch)
        )
        
        prompt = self._construct_prompt(
            molecule_pairs=molecule_pairs,
            n_per_molecule=n_per_molecule,
            strategy=strategy
        )
        print("\nPrompt sent to LLM:")
        print("-"*30)
        print(prompt)
        print("-"*30)
        
        print("\nGetting LLM response...")
        result = self._get_llm_response(prompt)
        print("\nLLM Response:")
        print("-"*30)
        print(result)
        print("-"*30)

        print("\nProcessing generations...")
        return self._process_generations(
            result=result,
            starting_smiles_list=starting_smiles_batch,
            target_conductivity=target_conductivity,
            n_per_molecule=n_per_molecule
        )
    
    def _process_generations(self, result, starting_smiles_list, target_conductivity, n_per_molecule):
        """Process LLM generations"""
        print("\nProcessing LLM Generations")
        print("-"*50)
        print(f"Number of parent molecules: {len(starting_smiles_list)}")
        print(f"Target conductivity: {target_conductivity:.2e} mS/cm")
        
        validated_results = {}
        metrics_results = {}
        
        for parent, generation_dict in result.generated_molecules.items():
            print(f"\nProcessing Parent: {parent}")
            print("-"*30)
            print(f"Number of generations: {len(generation_dict['smiles'])}")
            
            canonical_parent, metrics, valid_gens = self._process_parent_and_generations(
                parent, generation_dict, target_conductivity
            )
            
            if canonical_parent and valid_gens:
                print(f"\nSuccessful generations: {len(valid_gens)}/{n_per_molecule}")
                print("\nGenerated molecules:")
                for i, gen in enumerate(valid_gens, 1):
                    print(f"\n{i}. {gen['generated_smiles']}")
                    print(f"   Conductivity: {gen['actual_conductivity']:.2e} mS/cm")
                    print(f"   Parent Conductivity: {gen['parent_conductivity']:.2e} mS/cm")
                    print(f"   Improvement Factor: {gen['improvement_factor']:.2f}x")
                    print(f"   Explanation: {gen['explanation']}")
                
                validated_results[canonical_parent] = valid_gens
                if metrics:
                    print("\nMetrics:")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            print(f"{metric_name}: {value:.3f}")
                        else:
                            print(f"{metric_name}: {value}")
                    metrics_results[canonical_parent] = metrics
            else:
                print("No valid generations produced")
        
        print("\nGeneration Summary:")
        print("-"*30)
        print(f"Parents processed: {len(result.generated_molecules)}")
        print(f"Successful parents: {len(validated_results)}")
        if validated_results:
            improvements = [
                max(gens, key=lambda x: x['improvement_factor'])['improvement_factor']
                for gens in validated_results.values()
            ]
            print(f"Average improvement factor: {np.mean(improvements):.2f}x")
            print(f"Best improvement factor: {max(improvements):.2f}x")
        
        return result, validated_results, metrics_results
    
    def _process_parent_and_generations(self, parent, generation_dict, target_conductivity):
        """Process a parent and its generations"""
        try:
            canonical_parent = Chem.CanonSmiles(parent)
            parent_preds = predict([canonical_parent], 'conductivity', self.base_path)
            if not parent_preds:
                return None, None, []
                
            parent_conductivity = float(parent_preds[0]) * 1000
            
            valid_gens = []
            generated_smiles = []
            
            for gen_smiles, explanation in zip(generation_dict['smiles'], 
                                             generation_dict['reasoning']):
                result = self.process_single_generation(
                    gen_smiles, explanation, parent_conductivity, target_conductivity
                )
                if result:
                    valid_gens.append(result)
                    generated_smiles.append(result['generated_smiles'])
            
            metrics = None
            if generated_smiles:
                metrics = self.metrics_calculator.evaluate_single_generation(
                    parent_smiles=canonical_parent,
                    generated_smiles=generated_smiles,
                    explanations=generation_dict['reasoning']
                )
            
            return canonical_parent, metrics, valid_gens
            
        except Exception as e:
            print(f"Error processing parent {parent}: {str(e)}")
            return None, None, []
        
    def _get_llm_response(self, prompt: str, schema=MoleculeGeneration):
        """
        Get response from LLM using langfun query
        
        Args:
            prompt: The prompt to send to LLM
            schema: Schema to use for response (default: MoleculeGeneration)
        """
        model = LLMConfig.get_model(
            model_name=self.model_name,
            api_keys=self.api_keys,
            temperature=self.temperature
        )
        
        return lf.query(
            prompt=prompt,
            schema=schema,  # Use provided schema
            lm=model
        )