class PromptTemplates:
    """Templates for LLM prompts for polymer generation"""
    
    # Simplest possible initial strategy
    INITIAL_STRATEGY = """Generate valid structures"""
    
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
    def get_strategy_prompt(cls, previous_strategy: str, history_str: str, avg_improvement: float) -> str:
        """
        Construct strategy analysis prompt for strategy optimization
        
        Args:
            previous_strategy: Previous generation strategy
            history_str: String containing generation history
            avg_improvement: Average improvement factor
        """
        return f"""Analyze the generation results and propose an improved strategy to reach target ion conductivity.

Previous Strategy:
{previous_strategy}

Results (Average Improvement = {avg_improvement:.2f}x):
{history_str}

Based on these results:
1. What worked in the previous strategy?
2. What didn't work and why?
3. How can the strategy be improved?

Provide response in this format:
{{
    "strategy": "Clear steps for generation...",
    "analysis": {{
        "strengths": [
            "What worked well",
            "Why it worked"
        ],
        "weaknesses": [
            "What didn't work",
            "Why it failed"
        ],
        "recommendations": [
            "How to improve",
            "What to focus on"
        ]
    }}
}}

Focus on developing a better generation strategy based on the observed results."""


    @classmethod
    def construct_generation_prompt(cls,
                                  molecule_pairs: str,
                                  n_per_molecule: int,
                                  strategy: str,
                                  feedback: str = None) -> str:
        """
        Construct generation prompt
        
        Args:
            molecule_pairs: String of SMILES and target conductivities
            n_per_molecule: Number of generations per parent molecule
            strategy: Strategy to follow for generation
            feedback: Optional feedback from previous generation
            
        Returns:
            Complete prompt string for LLM
        """
        prompt = f"""{cls.TASK_DESCRIPTION}

For each of these parent polymer SMILES strings, generate {n_per_molecule} valid new polymers.

Parent SMILES and Target Conductivities:
{molecule_pairs}

Strategy to follow:
{strategy}"""

        if feedback:
            prompt += f"\n\nPrevious Generation Feedback:\n{feedback}"
        
        prompt += f"\n\n{cls.FORMAT_REQUIREMENT}"
        
        return prompt