# Research Experiments
## Paper "Automated Machine Learning to Enhance Knowledge Retrieval in Retrieval-Augmented Generation Pipelines"
### Paper accepted at HC@AIxIA 2025, co-located with ECAI 2025
📝 [link.springer.com/chapter/10.1007/978-3-032-16708-8_34](https://link.springer.com/chapter/10.1007/978-3-032-16708-8_34)

---

## Important!

### These are not the droids you are looking for...

![image](figures/sw-meme.png)

If you are looking for the experiments and results of the paper 

"Automated Machine Learning for Optimized Knowledge Retrieval:
Enhancing the Trustworthiness of Clinical RAG Pipelines", currently submitted in the journal Artificial Intelligence in Medicine

you can find them in this repository at the branch `special-issue` or by clicking [here](https://github.com/MatteoMagnini/experiments-2025-hcaixia-automl-rag/tree/special-issue).

---

## How to cite this paper

```bibtex
@inproceedings{DBLP:conf/hc/MagniniASMBDM25,
  author       = {Matteo Magnini and
                  Gianluca Aguzzi and
                  Leonardo Sanna and
                  Simone Magnolini and
                  Patrizio Bellan and
                  Mauro Dragoni and
                  Sara Montagna},
  editor       = {Pierangela Bruno and
                  Francesco Calimeri and
                  Francesco Cauteruccio and
                  Mauro Dragoni and
                  Fabio Stella and
                  Giorgio Terracina},
  title        = {Automated Machine Learning to Enhance Knowledge Retrieval in Retrieval-Augmented
                  Generation Pipelines},
  booktitle    = {Artificial Intelligence for Healthcare, and Hybrid Models for Coupling
                  Deductive and Inductive Reasoning - First International Joint Conference,
                  HC@AIxIA+HYDRA 2025, Bologna, Italy, October 25-26, 2025, Proceedings},
  series       = {Communications in Computer and Information Science},
  volume       = {2830},
  pages        = {429--440},
  publisher    = {Springer},
  year         = {2025},
  url          = {https://doi.org/10.1007/978-3-032-16708-8\_34},
  doi          = {10.1007/978-3-032-16708-8\_34},
  timestamp    = {Thu, 21 May 2026 17:36:52 +0200},
  biburl       = {https://dblp.org/rec/conf/hc/MagniniASMBDM25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

---

## Repository structure

- `chroma/`: Contains the code for the Chroma vector database used in the RAG pipeline.
- `data/`: Contains datasets used for training and evaluation.
- `documents/`: Contains documents used for retrieval in the RAG pipeline.
- `experiments/`: Contains scripts for running experiments and evaluating results.
- `figures/`: Contains scripts for generating figures and visualizations reported in the paper.
- `models/`: Contains code for training and evaluating machine learning models used in the paper.
- `results/`: Contains results from experiments, including metrics and logs.
- `utils/`: Contains utility functions.

---

## How to run the code

Be sure to have Python 3.12 or higher installed, and Poetry for dependency management.

1. Clone the repository:
```bash
git clone
```

2. Install dependencies:
```bash
poetry install
```

3. Run experiments:
```bash
python -m experiments
```