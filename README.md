# Distance Deflation
## Overview
This repository implements the algorithm presented in **[Euclidean Distance Deflation under High-Dimensional Heteroskedastic Noise](https://arxiv.org/abs/2507.18520)**

The algorithm is designed to estimate noise magnitudes and correct pairwise distances in high-dimensional datasets corrupted by heteroskedastic noise. 

Method Highlights: 
- Hyperparameter-free
- Requires no prior knowledge of the clean data structure or noise distribution
- No restrictive assumptions

## Algorithmic Workflow
<p align="center">
  <img src="Algorithmic_Workflow.png" width="70%">
</p>

## Installation
Install the package directly from GitHub:
```bash
pip install git+https://github.com/Keyi-Li/Distance_Deflation.git
```

## Demo
Run the demo in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keyi-Li/Distance_Deflation/blob/main/Demo.ipynb)
