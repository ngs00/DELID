# Self-Supervised Conditional Diffusion for Coarse-Graining Molecular Representation Learning

## Abstract
Electron-level information is an essential attribute determining the physical and chemical properties of molecules. However, the electron-level information is not accessible in most real-world molecules due to extensive computational costs to determine uncertain electronic structures. For this reason, existing methods for molecular representation learning have remained in the representation learning on simplified atom-level molecular descriptors. This paper proposes a molecular representation learning method based on a self-supervised conditional diffusion method to estimate the electron-level information about arbitrary complex real-world molecules from readily accessible fragmented information. The proposed method achieved state-of-the-art prediction accuracy on extensive real-world molecular datasets.


## Run
- train_src_model.py: A script to train the source models.
- calc_transferability.py: A script to calculate the transferability based on SCOTL.


## Datasets
The GWBC and HOIP-HSE datasets, which were used as the source calculation datasets, are available in [1] and [2], respectively.
The lists of the materials in the MPS and MPL datasets are presented in [the public repository of CGCNN](https://github.com/txie-93/cgcnn).
The crystal structures of all materials in the source calculations datasets are available at the Materials Project database [3].
All target experimental datasets are publicly available in their original papers, as presented in Table 1 of the paper.

**Dataset Repositories**
- GWBG dataset: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.115104
- HOIP-HSE dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.gq3rg
- MPS and MPL datasets: https://next-gen.materialsproject.org/
- EFE dataset: https://figshare.com/collections/Experimental_formation_enthalpies_for_intermetallic_phases_and_other_inorganic_compounds/3822835
- EU-TQT dataset: https://pubs.acs.org/doi/10.1021/acsami.9b16065
- TM dataset: https://github.com/ngs00/simd
- EBG dataset: https://pubs.acs.org/doi/10.1021/acs.jpclett.8b00124
- GTT dataset: https://figshare.com/articles/dataset/MAST-ML_Education_Datasets/7017254
