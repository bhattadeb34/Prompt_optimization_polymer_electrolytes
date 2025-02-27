# /noether/s0/dxb5775/htp_md/polymer_generation/prompts/evaluation_prompts.py

class EvaluationPromptTemplates:
    """Templates for evaluation prompts using best strategy"""
    
    TASK_DESCRIPTION = """Generate valid polymer SMILES that achieve target ion conductivity.

Example valid formats:
CC(CO[Cu])CSCCOC(=O)[Au]
CC(CN(C)CCOC(=O)[Au])N[Cu]
O=C([Au])NCCCCNC(=O)CCCN[Cu]"""
    
    FORMAT_REQUIREMENT = """Return your response in this exact format:
{
    "generated_molecules": {
        "<parent_smiles>": {
            "smiles": [
                "generated_smiles_1",
                "generated_smiles_2",
                ...
            ],
            "reasoning": [
                "explanation_1",
                "explanation_2",
                ...
            ]
        }
    }
}

IMPORTANT:
- Use proper JSON syntax
- All brackets and braces must be properly closed
- Lists must have matching number of items
- No trailing commas
- Use double quotes for strings
- Each parent must have exactly the requested number of generations

Each SMILES must have valid [Cu] and [Au] terminal groups in valid positions."""

    @classmethod
    def construct_evaluation_prompt(cls,
                                  molecule_pairs: str,
                                  n_per_molecule: int,
                                  best_strategy: str) -> str:
        """
        Construct prompt for evaluation using best strategy
        
        Args:
            molecule_pairs: String of parent SMILES and their target conductivities
            n_per_molecule: Number of generations per parent molecule
            best_strategy: Best strategy found from optimization
            
        Returns:
            Complete prompt string for evaluation
        """
        return f"""{cls.TASK_DESCRIPTION}

For each of these parent polymer SMILES strings, generate {n_per_molecule} valid new polymers.

Parent SMILES and Target Conductivities:
{molecule_pairs}

Use this optimized strategy that was found to be most effective:
{best_strategy}

{cls.FORMAT_REQUIREMENT}"""