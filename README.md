# README — Equilibrium Lattice Optimization with LAMMPS

## Overview

This workflow determines **equilibrium lattice parameters** by running LAMMPS energy minimizations from Python, iteratively adjusting box dimensions until the sum of absolute pressures approaches zero.

A **Python driver function** (`EquilibriumFunc`) controls the optimization process using `scipy.optimize.minimize`.
The **LAMMPS input template** (`20240817-Equilibrium.lammpstemp`) is parameterized to accept box dimensions, potential name, and output directory.

---

## Simulation Details

**Ensemble:**

* **NPT relaxation via `fix box/relax iso 0.0`** to minimize residual stress.
* Energy minimization performed with **conjugate gradient (CG)** until strict force and energy tolerances are met.

**Quantities Recorded:**

* Pressures (Pxx, Pyy, Pzz)
* Total energy, cohesive energy
* Lattice constant
* Volume, mass, density

---

## Workflow

1. **Prepare Input:**
   Python copies the LAMMPS template and replaces placeholders (`NumTemp`, `PotTemp`, `DirectoryTemp`, `BoxXTemp`, `BoxYTemp`, `BoxZTemp`).

2. **Run LAMMPS:**
   Executed via MPI (`mpirun -np 16`) using the specified binary.
   Output written to a log file and `Equilibrium.csv`.

3. **Extract Results:**
   Python reads `Equilibrium.csv` and calculates `PressureSum = |Pxx| + |Pyy| + |Pzz|`.

4. **Optimization:**
   `scipy.optimize.minimize` iteratively calls the function until convergence.

---

## Requirements

**Python Packages:**

* `pandas` — reading CSV output
* `scipy` — optimization (`Nelder-Mead`)
* `fileinput` & `shutil` — template handling
* `subprocess` — MPI execution

**External:**

* LAMMPS compiled with MPI
* `mpirun` or equivalent parallel execution command
* Template file `20240817-Equilibrium.lammpstemp`

---

## Output Files

* **Log**: `20240817-Equilibrium-{Num}.lammpslog`
* **CSV**: `Equilibrium.csv` — all recorded properties
* **Dump**: `Final.dump` — final atomic positions
* **Restart**: `Final.restart` — restart file
* Iteration history stored in `IterationDf`
