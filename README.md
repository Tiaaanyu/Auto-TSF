# Auto-TSF
![License](https://img.shields.io/badge/license-MIT-yellow)
![Issues](https://img.shields.io/github/issues/Tiaaanyu/Auto-TSF)
![Stars](https://img.shields.io/github/stars/Tiaaanyu/Auto-TSF)
![Forks](https://img.shields.io/github/forks/Tiaaanyu/Auto-TSF)
# Introduction
Code for paper: Auto-TSF: Towards Proxy-Model-based Meta-learning for Automatic Time Series Forecasting Algorithm Selection.
# Installation and Setup
## Prerequisites
- Python 3.x
- numpy >= 1.20.1
- sktime >=0.34.0
- multiprocessing >= 2.6.2.1
- pandas >= 2.2.3
## Running the project
1. Clone the repository:
```bash
   git clone https://github.com/Tiaaanyu/Auto-TSF.git
   cd Auto-TSF
```
2. Run locally:
```bash
  python automl.py
```
# Example usage
```python
    # data
    data_path = 'AirPassenger.csv'
    # Automatic algorithm selection
    alg = alg_selection(data_path)
    # Hyperparamete Optimization
    smape, hp = hpo(data_path, alg)
```
