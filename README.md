# Learning Coalition Value Functions from Relational Data

A graph neural network approach to non-superadditive international alliance games.

## Overview

This repository contains the code accompanying the paper:

> **Learning Coalition Value Functions from Relational Data: A Graph Neural Network Approach to Non-Superadditive International Alliance Games.**

We learn a coalition value function v: 2^N → R over the set of nation states from temporal heterogeneous relational data (alliances, trade, UN General Assembly voting, conflict) using a Relational Graph Convolutional Network (RGCN), and analyse the learned characteristic function with classical cooperative-game-theoretic tools: Shapley values, blocking analysis, the Nash Bargaining Solution, and the nucleolus / least-core. The learned game is systematically **non-superadditive** across every major coalition we examine.

## Headline Result

**Across all six major coalitions examined (NATO-26, EU-27, BRICS-5, Warsaw Pact ×2, ASEAN-10), the classical least-core is empty (ε > 0).** No transferable-utility allocation can simultaneously satisfy all sub-coalitions' rationality constraints, confirming non-superadditivity in a strong sense.

## Contributions

1. **A data-driven characteristic function.** v: 2^N → R obtained from a temporal heterogeneous graph via RGCN, evaluated on raw logits to avoid sigmoid saturation in the marginal-contribution regime.
2. **Monte-Carlo Shapley analysis** of historical coalitions, recovering both expected leadership orderings (USSR dominant in Warsaw Pact; USA, Germany, France leading NATO) and counter-intuitive findings (Hungary's Shapley value turning negative in the Warsaw Pact between 1980 and 1989).
3. **Blocking analysis** identifying natural "core pairs" and "core triplets" via marginal value gaps (e.g. NTH–FRN within NATO; THI–PHI–INS within ASEAN, three of the 1967 Bangkok founding members).
4. **Non-superadditivity spectrum** quantified by the blocking percentage, ranging from 26.6% (NATO-26, most stable) to 53.7% (ASEAN-10, most fragile).
5. **Comparative solution-concept analysis.** Shapley, Nash Bargaining Solution (marginal-threat variant), and nucleolus (exact LP for n ≤ 8; constraint-sampled LP for n > 8) are shown to answer three distinct questions ("average marginal contribution", "critical pivot", and "stability-preserving allocation" respectively), and can disagree in sign for transitional members.
6. **Temporal early-warning finding.** The blocking percentage rises sharply for the Warsaw Pact between 1980 and 1989 (25.6% → 37.0%) while the aggregate v(N) trajectory is nearly flat. **Internal inconsistency, not aggregate value, anticipates collapse.**

## Game-Theoretic Findings

### Empty cores across all six coalitions

| Coalition           | n  | v(N) logit | ε     | Core empty | Method                        |
|---------------------|----|------------|-------|------------|-------------------------------|
| BRICS 2014          | 5  | +3.81      | +2.16 | Yes        | exact LP                      |
| Warsaw Pact 1980    | 8  | +3.05      | +1.84 | Yes        | exact LP                      |
| Warsaw Pact 1989    | 8  | +2.93      | +1.62 | Yes        | exact LP                      |
| ASEAN-10 2010       | 10 | +2.59      | +2.27 | Yes        | sampled LP (800 constraints)  |
| NATO-26 2010        | 26 | +2.89      | +2.61 | Yes        | sampled LP (800 constraints)  |
| EU-27 2010          | 27 | +2.71      | +2.55 | Yes        | sampled LP (800 constraints)  |

### Three solution concepts disagree on a transitional member

Hungary, Warsaw Pact 1989 (raw logits):

| Concept                | Hungary | Reading                                        |
|------------------------|---------|------------------------------------------------|
| Shapley                | −0.36   | average marginal contribution                  |
| NBS (marginal threat)  | +0.62   | critical pivot                                 |
| Nucleolus              | −0.48   | stability-preserving allocation                |

The disagreement is interpretable: Hungary's role in the 1989 reform transition looks structurally different under each conceptual lens.

### Complementary fragility metrics

The blocking percentage and the least-core ε measure *different* aspects of non-superadditivity:

| Coalition       | % blocking         | ε                   | Interpretation                              |
|-----------------|--------------------|---------------------|---------------------------------------------|
| NATO-26 2010    | 26.6% (lowest)     | +2.61 (highest)     | Few but deep blocking pairs (NTH–FRN gap)   |
| ASEAN-10 2010   | 53.7% (highest)    | +2.27               | Widespread but individually shallow blocks  |

Reporting both metrics is required: a coalition may appear fragile in only one of them.

## Repository Structure

The pipeline is organised as sequential Jupyter notebooks. Notebooks 01–09 build the GNN model that serves as the characteristic function v(S); notebooks 10–11 carry out the cooperative-game-theoretic analysis on top of it.

| Notebook | Purpose |
|----------|---------|
| [01_download_data_v2.ipynb](01_download_data_v2.ipynb) | Download raw data from 9 sources |
| [02_preprocessing.ipynb](02_preprocessing.ipynb) | Raw data → canonical tables → PyG snapshots |
| [02b_fix_normalization.ipynb](02b_fix_normalization.ipynb) | z-score normalisation and snapshot rebuild |
| [03_coalition_samples.ipynb](03_coalition_samples.ipynb) | Positive / negative coalition sample construction |
| [04_model.ipynb](04_model.ipynb) | Initial model (baseline reference) |
| [04a_pretrain.ipynb](04a_pretrain.ipynb) | GNN encoder self-supervised pre-training |
| [04b_temporal.ipynb](04b_temporal.ipynb) | Temporal attention experiment (abandoned) |
| [05_ablation.ipynb](05_ablation.ipynb) | Baselines (LR / RF / MLP) and component ablations |
| [06_redesign.ipynb](06_redesign.ipynb) | Simplified GNN + rich pool (intermediate) |
| [07_full_gnn_rich.ipynb](07_full_gnn_rich.ipynb) | **Final model** v(S) learner (Test AUC 0.9384) |
| [08_case_studies.ipynb](08_case_studies.ipynb) | v(S) trajectories: NATO, EU, Warsaw Pact, ASEAN, BRICS |
| [09_error_analysis.ipynb](09_error_analysis.ipynb) | Confusion matrix, per-year / per-type / per-size analysis |
| [10_game_theory.ipynb](10_game_theory.ipynb) | **Shapley, blocking, NBS, nucleolus** on the learned v(S) |
| [11_experiments.ipynb](11_experiments.ipynb) | Extended game-theoretic experiments and robustness checks |

[data/](data/) contains:
- `canonical/`: cleaned input tables
- `snapshots/`: PyG temporal heterogeneous-graph snapshots
- `coalitions/`: positive and negative coalition samples
- `checkpoints/`: model checkpoints

## Methodology

### From relational data to a coalition value function

1. **Build a temporal heterogeneous graph** (1946–2016, approximately 180 country nodes per year) with four relation types: military alliances, trade, UN voting agreement, and armed conflict.
2. **Pre-train an RGCN encoder** by self-supervised masked feature reconstruction and link prediction.
3. **Fine-tune a coalition-validity head** that maps any subset S ⊆ N to a scalar via a rich-pool readout (mean + std + max + min of member embeddings and raw features, plus a bilinear pairwise term and a size feature).
4. **Define v(S) = logit_validity(S)** (the raw pre-sigmoid score), so that marginal contributions are not compressed by saturation.

### From v(S) to cooperative-game-theoretic quantities

- **Shapley values.** Monte-Carlo permutation sampling with antithetic variates.
- **Blocking analysis.** Enumerate sub-coalitions S' ⊂ S with v(S') > v(S) + δ; report the fraction (% blocking) and identify natural "core pairs" / "core triplets" via the largest marginal gaps.
- **Nash Bargaining Solution.** Marginal-threat variant: v(S) − v(S \ {i}) used as the threat point of player i.
- **Nucleolus / least-core ε.** Linear programming: exact for n ≤ 8 (full constraint set), constraint-sampled (800 random sub-coalitions) for n > 8.

## Underlying Model

The characteristic function v is learned by an RGCN encoder followed by a rich-pool coalition head trained on data from 1946 to 2009 with a 2010–2016 temporal hold-out (Test AUC 0.9384, F1 0.8218). See [07_full_gnn_rich.ipynb](07_full_gnn_rich.ipynb) for the architecture and training protocol; [05_ablation.ipynb](05_ablation.ipynb) contains baseline comparisons and component ablations.

## Reproduction

Notebooks were developed on Google Colab (A100 GPU). Recommended order:

1. Run notebooks [01](01_download_data_v2.ipynb)–[07](07_full_gnn_rich.ipynb) to fetch the data and train v(S).
2. Run [10_game_theory.ipynb](10_game_theory.ipynb) for Shapley, blocking, NBS, and nucleolus on the six headline coalitions.
3. Run [11_experiments.ipynb](11_experiments.ipynb) for extended experiments and robustness checks.
4. Run [08_case_studies.ipynb](08_case_studies.ipynb) for v(S) trajectories around historical regime changes.

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.4
- pandas, numpy, scipy, scikit-learn, matplotlib
- A linear-programming solver: `scipy.optimize.linprog` is sufficient for the nucleolus computations used here, and `cvxpy` is supported as an alternative.

## Citation

```bibtex
@article{learned_cooperative_games_2026,
  title   = {Cooperative Games with Learned Characteristic Functions: Axiomatic Foundations and an Application to International Coalitions},
  author  = {Kutsal, Mücahit},
  journal = {},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.

Copyright (c) 2026 Mücahit Kutsal.
