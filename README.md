# Physics-Informed Graph-Fourier Neural Operator (PI-GraphFNO) for Subsurface Flow
Author - Francis Lewis Obama Edzoa

A PyTorch implementation of a dual-branch **Graph-Fourier Neural Operator (PI-GraphFNO)** for transient two-phase porous-media flow under dynamically changing well counts and placements.

---

## Research Overview

In practical reservoir and geothermal field management, engineers continuously add infill wells, shut in high-water-cut producers, and modify injection schedules. Standard grid-based neural operators either require fixed well layouts or rely on sparse binary grid masks that ignore sub-grid well physics and violate fluid mass conservation.

### Core Research Question
> *When well counts and locations change at test time, does a hard mass-balance projection layer in a Graph-FNO achieve lower cell-level and global material-balance error (including well source/sink terms) than a soft finite-volume loss penalty?*

---

## Key Architectural Features

* **Spectral Field Backbone (`SpectralConv2d`):** Learns global spatial pressure and saturation dynamics across heterogeneous permeability fields using 2D Fast Fourier Transforms (FFT).
* **Dynamic Well Graph Encoder (`DynamicWellGraphEncoder`):** Represents active wells as permutation-invariant graph nodes encoding continuous normalized coordinates `(x, y)`, well type (`+1` injector / `-1` producer), bottom-hole pressure (`BHP`), and Peaceman Well Index (`WI`). Supports arbitrary well additions and shut-ins at inference without retraining.
* **Conservation Enforcement (`HardMassConservationLayer` vs. Soft Loss):**
  * **Baseline 1:** Purely data-driven Graph-FNO (MSE loss).
  * **Baseline 2:** Soft-constrained Graph-FNO with an additive finite-volume mass-balance residual loss.
  * **Proposed Model:** Hard-constrained Graph-FNO with a differentiable global volume-balance projection layer ensuring zero artificial fluid creation or loss.

---

## Quick Start

### 1. Requirements
* Python 3.9+
* PyTorch (`torch >= 2.0`)

```bash
pip install torch
```

### 2. Run the Baseline Verification
Execute `graph_fno_baseline.py` to test forward inference and mass-balance projection across variable well topologies (e.g., transitioning from a 3-well pattern to a 5-well infill pattern without changing network weights):

```bash
python graph_fno_baseline.py
```

---

## Repository Structure

```text
pi-graph-fno-reservoir/
├── graph_fno_baseline.py   # Core PI-GraphFNO architecture, graph well encoder, and mass-balance layers
└── README.md               # Project overview, research question, and usage instructions
```

---

## Roadmap & Future Extensions

* [ ] **Benchmark Dataset Generation:** Automated 2D heterogeneous two-phase waterflooding ensembles generated via Open Porous Media (OPM) Flow and ECLIPSE.
* [ ] **Ablation Study:** Quantitative comparison of Data-Only vs. Soft-Penalty vs. Hard-Projection Graph-FNO across out-of-distribution well counts.
* [ ] **Multiphysics & Geothermal Extension:** Extending the finite-volume conservation layer to coupled thermal-hydraulic (TH) energy transport for Enhanced Geothermal Systems (EGS).

