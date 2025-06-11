# Topological Data Analysis of Neural Network Optimizers

This project performs comprehensive topological data analysis (TDA) of neural network optimization trajectories and activations under gradient orthogonalization and spectral filtered optimizers.

## Overview

**Research Question**: How do different optimizers affect the topological structure of neural network activations during training?

**Approach**: 
- Train ensembles of MLPs on MNIST with different optimizers
- Run Mapper on top-50 PCA components of training trajectory
- Extract penultimate layer activations at key epochs
- Apply Vietoris-Rips persistent homology analysis
- Compare landscapes across optimizers


## Installation & Setup

### Prerequisites
- **Conda** or **Miniconda**
- **CUDA-capable GPU**
- **8GB+ RAM**

### Environment Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd tda_opt
```

2. **Create conda environment**:
```bash
conda env create -f environment.yml
```

3. **Activate environment**:
```bash
conda activate tda-opt
```

4. **Verify installation**:
```bash
python -c "import gudhi; import torch; import persim; print('All dependencies installed successfully!')"
```

### Troubleshooting Installation

If you encounter issues:

- **GUDHI fails**: Try `conda install -c conda-forge gudhi`
- **PyTorch GPU issues**: Install PyTorch manually for your CUDA version
- **Apple Silicon**: Use `conda install pytorch torchvision -c pytorch-nightly`
- **persim errors**: Try `pip install persim --no-deps`

## Datasets

- **MNIST**: Automatically downloaded on first run
- **MNIST-C**: Corrupted version generated with random affine transforms (±20° rotation, ±10% translation, scaling [0.9,1.1]) plus Gaussian noise
- **Location**: `data/MNIST/`

## Usage
### Training
Will take up to 8 hours, and requires 20GB free for storage.
```bash
run scripts/train_models.sh
```

### Activation Topology
#### Extract Activations
```bash
python scripts/run_activation_extraction.py
```

#### Compute Peristence Diagrams
```bash
python scripts/run_vietoris_rips_computation.py
```

#### Compute Persistent Landscapes
```bash
python analysis/analyze_landscapes.py
```

### Parameter Trajectory Topology
### Projections
```bash
python utils/projections.py
```

### Compute Mapper Graphs
```bash
python analysis/analyze_mapper.py
```

## Project Structure

```
tda_opt/
├── analysis/                    # TDA analysis scripts
│   ├── create_vietoris_rips.py  # Persistence computation
│   ├── analyze_vietoris_rips.py # Betti curves & landscapes
│   └── analyze_mapper.py        # Parameter space analysis
├── models/                      # Neural network architectures
├── results/                     # Analysis outputs & CSV data
├── figures/                     # Generated plots
├── utils/                       # Helper functions
├── main.py                      # Training script
├── extract_tda_data_to_csv.py   # Main analysis pipeline
└── environment.yml              # Conda environment
```


## System Requirements

- **Training**: ~2-3 hours on modern GPU, ~8-12 hours on CPU
- **TDA Analysis**: ~1-2 hours
- **Memory**: 8GB+ RAM recommended for full analysis
- **Storage**: 20GB+ for complete results

## Dependencies

**Core Libraries**:
- `torch` - Neural network training
- `gudhi` - Persistent homology computation  
- `persim` - Persistence diagram analysis
- `kmapper` - Mapper algorithm implementation
- `scikit-learn` - Machine learning utilities
- `umap-learn` - Dimensionality reduction

**Analysis & Visualization**:
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Plotting
- `plotly` - Interactive visualizations
- `networkx` - Graph analysis


## License

This project will be licensed under the MIT License when finished