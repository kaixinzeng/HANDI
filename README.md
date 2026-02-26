# HANDI: Flow-consistent Identification under Sparse Sampling

## Overview

This repository provides the reference implementation for the paper **“Flow-consistent identification of governing equations from sparsely sampled measurements.”** It implements Hybrid Analytic Neural Dynamics Identification (HANDI), a flow-consistent framework for identifying continuous-time governing equations from sparsely sampled time-series data. The method avoids numerical differentiation or integration by learning system flow maps in a hybrid observable space that combines analytic basis functions with neural residual observables, enabling both robustness and interpretability. Using a Koopman-based formulation, HANDI recovers physically consistent dynamics even under severe temporal sparsity and limited data, and is validated on canonical nonlinear systems and real-world datasets. A demonstration video illustrating the effects of sparse temporal sampling and the intuition behind the HANDI approach is provided.

The complete demonstration video is available at: https://github.com/kaixinzeng/HANDI/releases/download/v1.0-supp-video/2026-02-26_164548_006.mp4



https://github.com/user-attachments/assets/c14a2bbe-9c53-4ecf-98cd-cb92b056a109



## System requirements
The experiments were tested on the following system:
- **Operating system**: Ubuntu 24.04
- **Python**: 3.13  
- **GPU**: NVIDIA RTX 4090
- **NVIDIA driver**: version compatible with CUDA 13.0  

**Framework requirements**:  
- **PyTorch**: built with CUDA 12.9

**Python dependencies**:  
All required Python libraries and their exact versions are specified in `requirements.txt`.

## Installation guide
The code has been tested with **Python 3.9** in an isolated conda environment.
An NVIDIA GPU with CUDA support is required to run the demo. 
CUDA drivers should be properly installed on the host system.
Typical installation time is approximately **5–10 minutes**, assuming a CUDA-capable NVIDIA driver is already installed.

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

We provide several example datasets in the repository (canonical systems, benchmark examples, and real-world systems).
Each example folder is self-contained.

To run a demo using the provided data:

```bash
# Example: canonical Duffing system
cd Canonical systems/Duffing
python HANDI.py \
  --data ./data/duff_train40.npy \
  --dt 0.4 \
  --device 1
```

After execution, the demo **outputs** include trained model weights (.pt), a record of optimal hyperparameters (.json), and the identified symbolic governing equations (.txt). Additional files store identified coefficients, error metrics, and diagnostic indicators for quantitative evaluation and verification.

A representative demo illustrating the HANDI workflow is provided via Code Ocean. The Code Ocean capsule includes the necessary environment, data, and execution scripts to run the demo without additional setup. Users can execute the demo directly within the Code Ocean environment and inspect the resulting identified dynamics and visualizations. The expected runtime for the demo is approximately **3 minutes** on a standard GPU-enabled desktop environment.

Code Ocean demo capsule:  
https://codeocean.com/capsule/9813712/tree

## Instructions for use
Detailed descriptions of the code structure, benchmark systems, and experimental settings 
are provided in **`read_me.pdf`**.

To apply HANDI to custom datasets, prepare time-series data in the same format as the provided examples
(e.g., state variables sampled at uniform time intervals).

Users need to specify:
- Path to the input data
- Sampling interval `dt`
- (Optional) training-related hyperparameters

This can be done either via command-line arguments:

```bash
python HANDI.py --data <PATH_TO_DATA> --dt <SAMPLING_INTERVAL>
```

or by modifying the configuration section in `HANDI.py`.

## License
This project is released under the **MIT License**.
