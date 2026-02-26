# HANDI: Flow-consistent Identification under Sparse Sampling

This repository provides the reference implementation for the paper:

**Flow-consistent identification of governing equations from sparsely sampled measurements**  

## System requirements
The demo and experiments were tested on the following system:
- **Operating system**: Ubuntu 24.04 LTS  
- **Python**: 3.9  
- **GPU**: NVIDIA RTX 4090 (tested for demo)  
- **NVIDIA driver**: version compatible with CUDA 13.0  

**Framework requirements**:  
- **PyTorch**: built with CUDA 12.9

## Installation guide
The code has been tested with **Python 3.9** in an isolated conda environment.
An NVIDIA GPU with CUDA support is required to run the demo. 
CUDA drivers should be properly installed on the host system.

```bash
# Clone the repository
git clone <REPO_URL>
cd <repo-folder>

# Create and activate conda environment
conda create -n proj python=3.9 -y
conda activate proj

# Install dependencies
pip install -r requirements.txt
```

## Demo
Detailed descriptions of the code structure, benchmark systems, and experimental settings 
are provided in **`README.pdf`**.

## Instructions for use
Detailed descriptions of the code structure, benchmark systems, and experimental settings 
are provided in **`README.pdf`**.

## Repository overview
- `HANDI.py`, `EDMD.py`, `SINDy.py`, 'gEDMD.py','SR3.py','WSINDy.py','PSE.py': implementations of HANDI and baseline methods  
- Canonical systems, benchmark examples, and real-world examples are organized by folders  
- Scripts for comparison, ablation studies, and plotting are included

## Reproducibility
- Full-scale experiments may require GPU resources and are intended for offline execution.

## License
This project is released under the **MIT License**.
