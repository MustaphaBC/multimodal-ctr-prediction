# Multimodal Click-Through Rate Prediction

A research prototype for the MM-CTR challenge. It combines item-image representations with user behavior sequences and uses a Deep Interest Network (DIN) style model to estimate click-through probability.

## Pipeline

1. **Item representation:** `task1-ctr.ipynb` builds visual features with CLIP and aligns them with behavioral item vectors.
2. **CTR prediction:** `task2-ctr.ipynb` prepares embeddings and trains/evaluates the behavior-aware prediction pipeline.

The project report is in `CTR.pdf`; `mm_ctr_architecture_diagram.png` illustrates the system. The original folder also contains `RAG-Qween.ipynb`, a separate question-answering prototype; its maintained repository is [RAG_Qwen_Project](https://github.com/MustaphaBC/RAG_Qwen_Project).

## Requirements and data

The notebooks use Python machine-learning libraries including PyTorch, Transformers, NumPy, pandas/Polars, Numba, and challenge-specific tooling. Exact versions and dataset paths depend on the notebook configuration.

The MM-CTR workflow expects the challenge data, item images, and prepared feature/embedding files. These large data and model artifacts are not included in the public repository. Set the paths in the notebook configuration to your local copies before running.

## Run

Open `task1-ctr.ipynb` and `task2-ctr.ipynb` in Jupyter and execute the cells in order after installing the notebook dependencies and preparing the input data. The notebooks are research workflows rather than a packaged command-line application.

## Scope

Datasets, generated embeddings, trained model files, and notebook outputs are excluded from version control.
