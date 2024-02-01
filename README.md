# Self-Supervised Conditional Diffusion for Coarse-Graining Molecular Representation Learning

## Abstract
Electron-level information is an essential attribute determining the physical and chemical properties of molecules. However, the electron-level information is not accessible in most real-world molecules due to extensive computational costs to determine uncertain electronic structures. For this reason, existing methods for molecular representation learning have remained in the representation learning on simplified atom-level molecular descriptors. This paper proposes a molecular representation learning method based on a self-supervised conditional diffusion method to estimate the electron-level information about arbitrary complex real-world molecules from readily accessible fragmented information. The proposed method achieved state-of-the-art prediction accuracy on extensive real-world molecular datasets.


## Run
- ``exec.py``: A script to train and evaluate DELID for a given dataset.


## Datasets
We employed nine benchmark molecular datasets constructed by real-world chemical experiments.
The benchmark molecular datasets were selected from well-known databases in molecular science and biology [1, 2, 3].
For comprehensive evaluations, we selected the benchmark molecular datasets from four different chemical applications: physicochemistry, toxicity, pharmacokinetics, and optics.
The characteristics of the benchmark molecular datasets are summarized in the paper.

<span style='color: red'>fdafdas</span>

**Dataset Repositories**
- GWBG dataset: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.115104
- HOIP-HSE dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.gq3rg
- MPS and MPL datasets: https://next-gen.materialsproject.org/
- EFE dataset: https://figshare.com/collections/Experimental_formation_enthalpies_for_intermetallic_phases_and_other_inorganic_compounds/3822835
- EU-TQT dataset: https://pubs.acs.org/doi/10.1021/acsami.9b16065
- TM dataset: https://github.com/ngs00/simd
- EBG dataset: https://pubs.acs.org/doi/10.1021/acs.jpclett.8b00124
- GTT dataset: https://figshare.com/articles/dataset/MAST-ML_Education_Datasets/7017254


## References
[1] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), 513-530.
[2] Wu, K., & Wei, G. W. (2018). Quantitative toxicity prediction using topology based multitask deep neural networks. Journal of chemical information and modeling, 58(2), 520-531.
[3] Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., De Veij, M., Félix, E., ... & Leach, A. R. (2019). ChEMBL: towards direct deposition of bioassay data. Nucleic acids research, 47(D1), D930-D940.
