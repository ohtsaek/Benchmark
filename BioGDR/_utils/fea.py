#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Build input for AttentiveFP
"""
Created on Thu Dec 24 02:32:34 2020

@author: deepchem
"""
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit.Chem.AtomPairs.Utils import NumPiElectrons
RD_PT = Chem.rdchem.GetPeriodicTable()
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from rdkit.Chem import PeriodicTable
import dgl
def atom_hybridization(atom):
    """One hot encoding for the hybridization of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    """
    
    return int(atom.GetHybridization())

def atom_partial_charge(atom):
    """Get Gasteiger partial charge for an atom.
    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.
    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return float(gasteiger_charge)
def is_h_acceptor(atom):
    """ Is an H acceptor? """

    m = atom.GetOwningMol()
    idx = atom.GetIdx()
    return idx in [i[0] for i in Lipinski._HAcceptors(m)]   
def is_h_donor(a):
    """ Is an H donor? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._HDonors(m)]


def is_hetero(a):
    """ Is a heteroatom? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._Heteroatoms(m)]
def explicit_valence(a):
    """ Explicit valence of atom """
    return a.GetExplicitValence()
def implicit_valence(a):
    """ Implicit valence of atom """

    return a.GetImplicitValence()
def n_valence_electrons(a):
    """ return the number of valance electrons an atom has """

    return RD_PT.GetNOuterElecs(a.GetAtomicNum())
def n_pi_electrons(a):
    """ returns number of pi electrons """

    return NumPiElectrons(a)

def degree(a):
    """ returns the degree of the atom """

    return a.GetDegree()
def formal_charge(a):
    """ Formal charge of atom """

    return a.GetFormalCharge()
def num_implicit_hydrogens(a):
    """ Number of implicit hydrogens """

    return a.GetNumImplicitHs()


def num_explicit_hydrogens(a):
    """ Number of explicit hydrodgens """

    return a.GetNumExplicitHs()


def n_hydrogens(a):
    """ Number of hydrogens """

    return num_implicit_hydrogens(a) + num_explicit_hydrogens(a)
def n_lone_pairs(a):
    """ returns the number of lone pairs assicitaed with the atom """

    return int(0.5 * (n_valence_electrons(a) - degree(a) - n_hydrogens(a) -
                      formal_charge(a) - n_pi_electrons(a)))
def crippen_log_p_contrib(a):
    """ Hacky way of getting logP contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][0]
def crippen_molar_refractivity_contrib(a):
    """ Hacky way of getting molar refractivity contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][1]
def tpsa_contrib(a):
    """ Hacky way of getting total polar surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcTPSAContribs(m)[idx]
def atomic_mass(a):
    """ Atomic mass of atom """

    return a.GetMass()
def labute_asa_contrib(a):
    """ Hacky way of getting accessible surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcLabuteASAContribs(m)[0][idx]
def van_der_waals_radius(a):
    """ returns van der waals radius of the atom """
    return PeriodicTable.GetRvdw(rdkit.Chem.GetPeriodicTable(),+a.GetAtomicNum())
def get_min_ring(a):
    mol_=a.GetOwningMol()  
    r=mol_.GetRingInfo()
    r_lst=r.AtomRings()
    min_ring=min([len(i) for i in r_lst])
    return min_ring
def get_min_ring(b):
    mol_=b.GetOwningMol()  
    r=mol_.GetRingInfo()
    r_lst=r.AtomRings()
    
    count_ring=[len(i) for i in r_lst]
    if count_ring==[]:
        min_ring=0
    else:
        min_ring=min(count_ring)
    return min_ring
def  featurize_atoms(mol):
    feats = []
    AllChem.ComputeGasteigerCharges(mol)
    num_atom=len(mol.GetAtoms())
    for atom in mol.GetAtoms():
        feats.append([atom.GetAtomicNum(),
                      atom_hybridization(atom),
                      num_explicit_hydrogens(atom),
                      num_implicit_hydrogens(atom),
                      formal_charge(atom),
                      atom_partial_charge(atom),
                      atom.GetNumRadicalElectrons(),
                      atom.GetIsAromatic(),
                      degree(atom),
                      get_min_ring(atom),
                      
                      is_h_acceptor(atom),
                      is_h_donor(atom),
                      is_hetero(atom),
                    explicit_valence(atom),
                    implicit_valence(atom),
                    n_valence_electrons(atom),
                    crippen_log_p_contrib(atom),
                    crippen_molar_refractivity_contrib(atom),
                    tpsa_contrib(atom),
                    atomic_mass(atom),
                     n_pi_electrons(atom),
                      n_valence_electrons(atom),
                       n_lone_pairs(atom),
                       labute_asa_contrib(atom),
                       van_der_waals_radius(atom),
                       int(atom.GetChiralTag())]
    )
    return {'hv': torch.Tensor(feats).reshape(num_atom, -1).float()}

def construct_bigraph_from_mol(mol, node_featurize):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    g.ndata.update(node_featurize(mol))
    # Add edges
    src_list = []
    dst_list = []
    bond_fea=[]
    for u in range(num_atoms):
        for v in range(num_atoms):
            
            b = mol.GetBondBetweenAtoms(u, v)
            if u != v  :
                bf=[len(Chem.rdmolops.GetShortestPath(mol, u, v))-1  ]
            else:
                bf=[0]
            if b is not None:
                ring_length=get_min_ring(b)
                bf.extend([int(b.GetIsConjugated()),int(b.GetBondType()),ring_length])
            else:
                bf.extend([0]*3)
    
            bond_fea.append(bf)
            src_list.extend([u])
            dst_list.extend([v])
    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))
    
    g.edata.update({'he':torch.Tensor(bond_fea)})
    return g
def construct_bigraph_from_smiles(smiles, node_featurize):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    g.ndata.update(node_featurize(mol))
    # Add edges
    src_list = []
    dst_list = []
    bond_fea=[]
    for u in range(num_atoms):
        for v in range(num_atoms):
            
            b = mol.GetBondBetweenAtoms(u, v)
            if u != v  :
                bf=[len(Chem.rdmolops.GetShortestPath(mol, u, v))-1  ]
            else:
                bf=[0]
            if b is not None:
                ring_length=get_min_ring(b)
                bf.extend([int(b.GetIsConjugated()),int(b.GetBondType()),ring_length])
            else:
                bf.extend([0]*3)
    
            bond_fea.append(bf)
            src_list.extend([u])
            dst_list.extend([v])
    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))
    
    g.edata.update({'he':torch.Tensor(bond_fea)})
    return g



if __name__=="__main__":
    
    mol = Chem.MolFromSmiles('C1CCCCC1')
    print(featurize_atoms(mol))
    g = construct_bigraph_from_mol(mol,featurize_atoms)
    print(g.ndata['hv'].shape)
    print(g.edata['he'].shape)


