# PATE Dynamic Synthesis

**Improving the Privacy of Federated Learning via Optimized Synthetic Data Generation**

Master Thesis – Felix Kempe, TU Wien
Supervised by Ao.Univ.Prof. Dr. Andreas Rauber

## Overview

This repository implements two dynamic synthesis strategies that close the *Unlabeled Data Gap* in the PATE framework under Non-IID conditions:

- **Variant A (Distribution-Aware Synthesis):** Inverse confidence weighting based on the student's class confidence map
- **Variant B (Error-Aware Synthesis):** Error-proportional weighting with confusion-based contrast pairs

The PATE aggregation logic and teacher CNN architecture are adapted from the NEWSROOM/Saferlearn Framework (Thales Research & Technology).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Run a single experiment (MNIST, β=0.1, Variant A)
python -m src.experiments.run_experiment --config configs/default.yaml \
    --dataset MNIST --beta 0.1 --strategy variant_a --seed 0

# Run full evaluation suite
python -m src.experiments.run_experiment --config configs/default.yaml
```

## Acknowledgments

The PATE aggregation logic and teacher CNN architecture are adapted from the NEWSROOM/Saferlearn framework (Romain Ferrari et al., Thales Research & Technology).
