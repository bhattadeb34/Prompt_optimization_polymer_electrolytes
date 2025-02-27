import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
import json
import sys
from io import StringIO
import math
import os
# Global variable for fpscores
_fscores = None

def check_novelty(df_generated, df_train, column_name):
    """Compare generated polymers with training set for novelty"""
    for i in df_generated[column_name]:
        if df_train[column_name].eq(i).any():
            df_generated.loc[df_generated[column_name] == i, 'diversity'] = 'In the original data set'
        else:
            df_generated.loc[df_generated[column_name] == i, 'diversity'] = 'novel'
    return df_generated

def validate_mol(mol_list, column_name):
    """Check polymer validity"""
    sio = sys.stderr = StringIO()
    for i in mol_list['mol_smiles']:
        if pd.isna(i):
            mol_list.loc[mol_list[column_name] == i, 'validity'] = "none"
        elif Chem.MolFromSmiles(i) is None:
            mol_list.loc[mol_list[column_name] == i, 'validity'] = sio.getvalue().strip()[11:]
            sio = sys.stderr = StringIO()  # reset error logger
        elif ('=[Cu]' in i) or ('[Cu]=' in i) or ('=[Au]' in i) or ('[Au]=' in i):
            mol_list.loc[mol_list[column_name] == i, 'validity'] = 'Double bond at the end point'
        elif ('#[Cu]' in i) or ('[Cu]#' in i) or ('#[Au]' in i) or ('[Au]#' in i):
            mol_list.loc[mol_list[column_name] == i, 'validity'] = 'Triple bond at the end point'
        elif (i.count("[Cu]") != 1) or (i.count("[Au]") != 1):
            mol_list.loc[mol_list[column_name] == i, 'validity'] = 'More than two ends'
        else:
            bond_flag = False
            for atom in Chem.MolFromSmiles(i).GetAtoms():
                if atom.GetSymbol() == "Cu":
                    if atom.GetDegree() > 1:
                        bond_flag = True
                elif atom.GetSymbol() == "Au":
                    if atom.GetDegree() > 1:
                        bond_flag = True
            if bond_flag:
                mol_list.loc[mol_list[column_name] == i, 'validity'] = 'More than one bonds at the end point'
            else:
                mol_list.loc[mol_list[column_name] == i, 'validity'] = 'ok'
    return mol_list

def has_two_ends(df):
    """Check if polymer has correct terminal groups"""
    for mol in df['mol_smiles']:
        if pd.isna(mol):
            continue
        elif (mol.count('[Cu]') == 1 and mol.count('[Au]') == 1):
            df.loc[df['mol_smiles'] == mol, 'has_two_ends'] = True
        else:
            df.loc[df['mol_smiles'] == mol, 'has_two_ends'] = False
    return df

def numBridgeheadsAndSpiro(mol, ri=None):
    """Calculate number of bridgehead and spiro atoms"""
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    """Calculate synthetic accessibility score"""
    global _fscores
    if _fscores is None:
        raise ValueError("_fscores not initialized")

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # transform "raw" score into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

def calculate_morgan_fingerprint(smiles_lst):
    """Calculate Morgan fingerprints for a list of SMILES"""
    radius = 2
    n_bits = 2048
    
    fp_lst = []
    for s in smiles_lst:
        mol = Chem.MolFromSmiles(s)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fp_lst.append(fingerprint)
    
    return fp_lst

def calculate_diversity(smiles_lst):
    """Calculate diversity score for a list of SMILES"""
    fp_lst = calculate_morgan_fingerprint(smiles_lst)
    diversity_lst = []
    
    for i in range(len(smiles_lst)):
        for j in range(i):
            similarity = TanimotoSimilarity(fp_lst[i], fp_lst[j])
            diversity_lst.append(1-similarity)
    return diversity_lst, np.mean(diversity_lst)


class PolymerMetrics:
    def __init__(self, fpscores_path: str = "fpscores.json"):
        """Initialize PolymerMetrics with path to fpscores.json"""
        global _fscores
        if _fscores is None:
            # Get the directory where polymer_metrics.py is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(current_dir, "fpscores.json")
            
            # Use provided path or default
            path_to_use = fpscores_path if os.path.exists(fpscores_path) else default_path
            
            with open(path_to_use, 'r') as f:
                data = json.load(f)
            outDict = {}
            for i in data:
                for j in range(1, len(i)):
                    outDict[i[j]] = float(i[0])
            _fscores = outDict

    def evaluate_single_generation(self, parent_smiles: str, generated_smiles: list[str], 
                                 explanations: list[str] = None) -> dict:
        """Evaluate metrics for a single parent's generations"""
        # Create DataFrames
        parent_df = pd.DataFrame({"mol_smiles": [parent_smiles]})
        gen_df = pd.DataFrame({"mol_smiles": generated_smiles})
        num_samples = len(generated_smiles)

        # Uniqueness
        gen_df['duplicate'] = gen_df['mol_smiles'].duplicated()
        uniqueness = 1 - len(gen_df[gen_df['duplicate'] == True]) / num_samples

        # Novelty
        gen_df = check_novelty(gen_df, parent_df, 'mol_smiles')
        count_not_novel = gen_df['mol_smiles'][gen_df['diversity'] != 'novel'].count()
        novelty = 1 - count_not_novel / num_samples

        # Validity
        gen_df_valid = validate_mol(gen_df, 'mol_smiles')
        gen_df_valid = has_two_ends(gen_df_valid)
        df_valid = gen_df_valid.loc[(gen_df_valid['validity'] == 'ok') & 
                                   (gen_df_valid['has_two_ends'] == True)]
        validity = len(df_valid) / num_samples

        # Clean DataFrame
        df_clean = gen_df_valid.loc[
            (gen_df_valid['duplicate'] == False) & 
            (gen_df_valid['diversity'] == 'novel') & 
            (gen_df_valid['validity'] == 'ok') & 
            (gen_df_valid['has_two_ends'] == True)
        ]

        metrics = {
            'uniqueness': uniqueness,
            'novelty': novelty,
            'validity': validity,
            'valid_smiles': df_valid['mol_smiles'].tolist()
        }

        if len(df_clean) > 0:
            # Synthesizability
            sa_scores = [calculateScore(Chem.MolFromSmiles(s)) for s in df_clean["mol_smiles"]]
            metrics['synthesizability'] = len([x for x in sa_scores if x < 5]) / len(df_clean)
            metrics['sa_scores'] = sa_scores

            # Similarity
            morgan_fingerprint_generated = calculate_morgan_fingerprint(df_clean["mol_smiles"])
            morgan_fingerprint_original = calculate_morgan_fingerprint([parent_smiles])

            tanimoto_similarity = []
            for i in range(len(df_clean["mol_smiles"])):
                f1 = morgan_fingerprint_generated[i]
                similarity_scores = [TanimotoSimilarity(f1, f2) for f2 in morgan_fingerprint_original]
                tanimoto_similarity.append(np.mean(similarity_scores))
            metrics['similarity'] = np.mean(tanimoto_similarity)

            # Diversity
            diversity_lst, diversity = calculate_diversity(df_clean["mol_smiles"].tolist())
            metrics['diversity'] = diversity
            metrics['diversity_list'] = diversity_lst

        return metrics