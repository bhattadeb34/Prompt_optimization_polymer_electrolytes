from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, Optional

class SMILESProcessor:
    """Utilities for processing SMILES strings"""
    
    @staticmethod
    def validate_polymer_smiles(smiles: str) -> Tuple[bool, str]:
        """Validate polymer SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES"
            
            if '[Cu]' not in smiles or '[Au]' not in smiles:
                return False, "Missing terminal groups"
                
            if any(x in smiles for x in ['=[Cu]', '[Cu]=', '=[Au]', '[Au]=', 
                                       '#[Cu]', '[Cu]#', '#[Au]', '[Au]#']):
                return False, "Invalid terminal bonds"
                
            return True, "Valid polymer SMILES"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def process_polymer_smiles(smiles: str) -> Optional[str]:
        """Process SMILES for polymer structure"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            mol = Chem.AddHs(mol)
            
            # Form ring
            rxn = AllChem.ReactionFromSmarts('([Cu][*:1].[*:2][Au])>>[*:1]-[*:2]')
            results = rxn.RunReactants([mol])
            
            if not (len(results) == 1 and len(results[0]) == 1):
                rxn = AllChem.ReactionFromSmarts('([Cu]=[*:1].[*:2]=[Au])>>[*:1]=[*:2]')
                results = rxn.RunReactants([mol])
                
            if not (len(results) == 1 and len(results[0]) == 1):
                return None
                
            result_mol = results[0][0]
            Chem.SanitizeMol(result_mol)
            
            return Chem.MolToSmiles(result_mol, canonical=True)
        except:
            return None