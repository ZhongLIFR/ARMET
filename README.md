# ARMET: Cross-Domain Graph-Level Anomaly Detection

[![Journal](https://img.shields.io/badge/Journal-TKDE%202024-blue.svg)](https://doi.org/10.1109/TKDE.2024.3462442)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTKDE.2024.3462442-b31b1b.svg)](https://doi.org/10.1109/TKDE.2024.3462442)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-1682d4.svg)](https://doi.org/10.5281/zenodo.8246213)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python](https://img.shields.io/badge/Python-environment.yml-blue.svg)](https://www.python.org/)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=ZhongLIFR%2FARMET&countColor=%23263759&style=flat)](https://visitorbadge.io/status?path=ZhongLIFR%2FARMET)

This is the official repository for the paper:

> **Cross-Domain Graph Level Anomaly Detection**  
> Published in **IEEE Transactions on Knowledge and Data Engineering (TKDE)**, 2024.  
> [DOI](https://doi.org/10.1109/TKDE.2024.3462442) | [DBLP](https://dblp.org/rec/journals/tkde/LiLSL24a) | [Dataset](https://doi.org/10.5281/zenodo.8246213)

ARMET addresses **cross-domain graph-level anomaly detection**: detecting anomalous graphs in an unlabeled target domain by leveraging easily accessible normal graphs from a different but related source domain.

This setting is motivated by two practical challenges. First, graph-level anomaly labels are often expensive or unavailable. Second, many unsupervised graph anomaly detection methods assume that training data contains only normal graphs, which can be fragile when the target set is contaminated by anomalies.

---

## Highlights

- **Cross-domain setting**: ARMET uses normal source-domain graphs to improve anomaly detection on an unlabeled target domain.
- **Graph-level anomaly detection**: The method identifies anomalous graphs from a collection of graphs, rather than anomalous nodes or edges inside a single graph.
- **Domain-invariant representation learning**: An adversarial domain classifier encourages graph representations that transfer across related domains.
- **Source-domain supervision**: A one-class classifier exploits the available normal-label information in the source domain.
- **Class alignment**: A class aligner uses pseudo-labels to improve alignment between source and target distributions.
- **Benchmark data release**: The datasets are provided through Zenodo and should be downloaded separately before running the code.

---

## Method Overview

ARMET is designed for the case where the target domain is unlabeled but a related source domain provides accessible normal graphs. At a high level, the model contains four main components:

1. **Feature extractor**: learns graph-level representations that preserve semantic and topological information while incorporating distances between graphs.
2. **Adversarial domain classifier**: encourages source and target graph representations to be domain-invariant.
3. **One-class classifier**: uses normal source-domain graphs to learn a compact representation of normality.
4. **Class aligner**: aligns source and target classes using pseudo-labels, improving transfer to the target domain.

Together, these components aim to transfer useful normality information from the source domain while adapting to the unlabeled target-domain distribution.

---

## Repository Structure

```text
ARMET/
├── ComVis.pdf          # Visualization / supplementary material
├── DataLoader.py       # Data loading utilities
├── environment.yml     # Conda environment specification
├── main.py             # Main entry point for experiments
└── README.md           # Project documentation
```

The dataset is not stored directly in the repository. Please download it from Zenodo as described below.

---

## Setup

### Step 0: Check Requirements

Create the environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ARMET
```

If your local environment name differs from `ARMET`, please use the environment name specified in `environment.yml`.

---

## Dataset

Download `Data.zip` from Zenodo:

[https://doi.org/10.5281/zenodo.8246213](https://doi.org/10.5281/zenodo.8246213)

After downloading, unzip it and place the resulting folder under the repository root:

```text
ARMET/
├── Data/
├── DataLoader.py
├── environment.yml
└── main.py
```

Please make sure the folder is named exactly:

```text
Data
```

If the unzipped folder has a different name, rename it to `Data`.

---

## Running Experiments

Run the main experiment script from the repository root:

```bash
python main.py
```

Full experiments may take a long time depending on hardware resources. The original README notes that reproducing the full experimental results can require several weeks, or even several months, on limited computational resources.

For a quick sanity check, we recommend first running a reduced configuration before launching the full benchmark.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{li2024cross,
  title={Cross-Domain Graph Level Anomaly Detection},
  author={Zhong Li and Sheng Liang and Jiayang Shi and Matthijs van Leeuwen},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={36},
  number={12},
  pages={7839--7850},
  year={2024},
  doi={10.1109/TKDE.2024.3462442},
  url={https://doi.org/10.1109/TKDE.2024.3462442}
}
```

---

## License

This repository is released under the **CC BY-SA 4.0** license.

---

## Contact

For questions, bug reports, or suggestions, please open an issue in this repository.
