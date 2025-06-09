# TraceHiding: A Machine Unlearning Framework for Trajectory Classification

This repository contains the official implementation of the **TraceHiding** framework, developed as part of the thesis:

> **"TraceHiding: An Algorithmic Framework for Machine Unlearning in Mobility Data"**
> *Ali Faraji, 2025*
> (Submitted to *ACM Transactions on Spatial Algorithms and Systems*)

## 🧠 Overview

As machine learning models increasingly rely on personal and location-based data, **user privacy** has become a critical concern. This work introduces a novel framework for **machine unlearning** in the context of **trajectory classification**—an area where removing learned representations poses significant challenges due to the spatiotemporal nature of the data.

**TraceHiding** offers an efficient and effective solution for selectively forgetting user trajectories from trained models without costly retraining.

## ✨ Key Features

* 📍 **Trajectory Unlearning**: Designed specifically for trajectory-based data.
* ⚖️ **Privacy–Utility Balance**: Optimizes privacy while preserving predictive performance.
* 📊 **Influence Ranking**: Uses a hierarchical scoring system to rank tokens, trajectories, and users by their influence on the model.
* 🧪 **Distillation Loss**: Applies a teacher–student distillation technique to retain useful patterns while forgetting specific data points.
* 📈 **Reproducible Benchmarks**: Includes datasets, train/test splits, and evaluation tools to support reproducibility and comparison.

## 📂 Repository Structure

> [!TIP]
> you can browse the repository and explore in the directories. I will provide detailed explanation of each file and directory soon.

## 🚀 Getting Started

### Prerequisites

> [!NOTE]  
> I will clean up the requirements and put the commands here to create the env by `environment.yml`

### Running an Experiment

> [!NOTE]  
> A man page documentaion will be provided.

Preprocessed datasets are available on Zenodo.

## 📚 Publications

If you use this code or ideas from this work, please cite:

```bibtex
@article{faraji2025tracehiding,
  author = {Faraji, Ali and Papagelis, Manos},
  title = {TraceHiding: An algorithmic framework for machine unlearning in mobility data},
  year = {2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  journal = {ACM Transactions on Spatial Algorithms and Systems},
  note = {Submitted}
}
```

Related publication:

```bibtex
@inproceedings{faraji2023point,
  author = {Faraji, Ali and Li, Jing and Alix, Gian and Alsaeed, Mahmoud and Yanin, Nina and Nadiri, Amirhossein and Papagelis, Manos},
  title = {Point2Hex: Higher-order Mobility Flow Data and Resources},
  year = {2023},
  isbn = {9798400701689},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3589132.3625619},
  doi = {10.1145/3589132.3625619},
  booktitle = {Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems},
  articleno = {69},
  numpages = {4},
  keywords = {trajectory datasets, higher-order mobility flow datasets, generator},
  location = {Hamburg, Germany},
  series = {SIGSPATIAL '23}
}
```


## 🙋‍♀️ Contributing

We welcome contributions, issues, and pull requests!

## 🔍 Acknowledgments

This work was conducted at the intersection of geospatial computing and machine learning privacy, and is deeply indebted to the research and insights shared by the academic and open-source communities.


