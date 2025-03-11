# DOME
This repository contains the code and data for "DOME: Directional Medical Embedding Vectors from Electronic Health Records"


The increasing availability of electronic health record (EHR) systems has created enormous potential for translational research. Representative learning algorithms have provided an effective approach to obtaining numeric representations of EHR concepts. However, existing methods for learning embedding vectors suffer from two shortcomings. They neglect the longitudinality of the EHR data and often require patient-level data. The former overlooks the importance of temporal information for clinical researches such as drug side effects discovery and disease-onset prediction while the latter is against the principle of preserving patients' privacy and hinders data sharing across institutions. 

We introduce DirectiOnal Medical Embedding (DOME) vectors, which resolve those issues by learning the temporally directional relationships between medical concepts from aggregated data only. Specifically, DOME first computes two summary matrices, based on aggregated patient-level EHR data, capturing the information of the past and the future, respectively. Then, joint matrix factorization is performed on the two matrices to distill three vectors for each concept, consisting of a semantic embedding and two directional context embeddings correspondingly for the past and the future. These three embeddings together comprehensively describe the temporally directional dependency between concepts

# Requirements
* python 3.7
* numpy >= 1.19
* pandas >= 1.3
* scikit-learn >= 1.0.2

# Usage
```sh
  1. python PPMI.py
  2. python assym_word_embed.py
```
