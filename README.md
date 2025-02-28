# Self-Supervised Diffusion Models for Electron-Aware Molecular Representation Learning
Physical properties derived from electronic distributions are essential information that determines molecular properties. However, the electron-level information is not accessible in most real-world complex molecules due to the extensive computational costs of determining uncertain electronic distributions. For this reason, existing methods for molecular property prediction have remained in regression models on simplified atom-level molecular descriptors, such as atomic structures and fingerprints. This paper proposes an efficient knowledge transfer method for electron-aware molecular representation learning. To this end, we devised a self-supervised diffusion method that estimates the electron-level information of real-world complex molecules without expensive quantum mechanical calculations. The proposed method achieved state-of-the-art prediction accuracy in the tasks of predicting molecular properties on extensive real-world molecular datasets.

# Before Execution
- Please check the ``requirements.txt`` file before the execution of HEDMoL.
- You can download most benchmark molecular datasets at https://moleculenet.org/datasets-1.

# Run
You can train and evaluate DELID by executing ``exec.py``.

# Notice
- Please check the hyperparameter settings in Appendix F of the paper.
- An example Lipop dataset in this project is provided in [1].

# Reference
[1] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), 513-530.
