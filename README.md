# LAMMPS MTP Potential Runner

This repository contains Python code to run LAMMPS simulations with Moment Tensor Potentials (MTP) using MPI on an HPC environment or locally on Windows.

## Description
The code automates the execution of LAMMPS with provided input scripts and potential files. It uses MPI for parallel execution on Linux clusters and can also run with a locally installed LAMMPS (with Python integration) on Windows.

The environment is automatically detected:
- **Windows**: Runs locally using the installed LAMMPS Python module
- **Linux**: Runs on HPC systems such as Compute Canada with MPI and SLURM

## Requirements
- Python 3
- LAMMPS compiled with MPI support (Linux) or with Python integration (Windows)
- SLURM workload manager (for HPC usage)
- Installed Python packages:
  - numpy
  - pandas
  - scipy

## HPC Environment (Linux)
Required modules:
```bash
module load gcc
module load openmpi
