import pyglove as pg

class MoleculeGeneration(pg.Object):
    """Schema for batch polymer generation with reasoning"""
    generated_molecules: dict[str, dict[str, list]]


class StrategyResponse(pg.Object):
    """Schema for strategy analysis response"""
    strategy: str  # Single string containing the strategy
    analysis: dict[str, list[str]]  # Dict with strengths, weaknesses, recommendations