# Distance Deflation
## Overview
This repository implements the algorithm from: **[Euclidean Distance Deflation under High-Dimensional Heteroskedastic Noise](https://arxiv.org/abs/2507.18520)**

The algorithm is designed to estimate noise magnitudes and correct pairwise distances in high-dimensional datasets corrupted by heteroskedastic noise. 

Method Highlights: 
- hyperparameter-free
- require no prior knowledge of the clean data structure or noise distribution
- no restrictive assumption

The workflow is as follows: <p align="center">
  <img src="Algorithmic_Workflow.png" width="70%">
</p>

## Installation
The package can be installed by
```bash
pip install git+https://github.com/Keyi-Li/Distance_Deflation.git
