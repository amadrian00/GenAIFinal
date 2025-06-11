# Evaluating the Evolution of Graph Generative Models: Autoregressive, VAE, GAN and Diffusion Approaches

 

## Overview

This project compares different generative artificial intelligence methods for graphs, applied in the particular case of molecule generation.
The model's code is almost equal to the original one in its respective repository, just minimal changes have been made.
- GraphRNN [1]: https://github.com/JiaxuanYou/graph-generation
- CCGVAE [2]: https://github.com/drigoni/ConditionalCGVAE/tree/master
- MolGAN [3]: https://github.com/nicola-decao/MolGAN/tree/master (original paeper) 
  - https://keras.io/examples/generative/wgan-graphs/ (code basis)
- DiGress [4]: https://github.com/cvignac/DiGress/tree/main

---

## Project Structure
```
├── models/
    ├── __init__.py
│   ├── ConditionalCGVAE-master     # CCGVAE model
│   ├── DiGress-main                # DiGress model
│   ├── graph-generation-master     # GraphRNN model
│   └── MolGAN                      # MolGAN model
│
├── main.py                         # Main script to perform evaluation of the models.
├── metrics.py                      # Definition of the validity, uniqueness and novelty metrics.
└── README.md                       # Project documentation
```

---

## Features

- **Multiple GNN autoencoder models**: Compare ChebNet, GAT, and GCN based autoencoders.
- **Flexible preprocessing**: Includes graph data preparation and feature engineering.
- **Training and evaluation**: Easily train models and evaluate performance on given datasets.
- **Embedding projection**: Additional embedding manipulation via the projector module.

---

## Usage
- Running `main.py` is enough to run all the training and the test battery.

## Contact
amadrian@korea.ac.kr

## References
[1]	J. You, R. Ying, X. Ren, W. Hamilton, and J. Leskovec, “GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models,” in Proceedings of the 35th International Conference on Machine Learning, PMLR, Jul. 2018, pp. 5708–5717.

[2]	D. Rigoni, N. Navarin, and A. Sperduti, “Conditional constrained graph variational autoencoders for molecule design,” in 2020 IEEE symposium series on computational intelligence (SSCI), IEEE, 2020, pp. 729–736.

[3]	N. D. Cao and T. Kipf, “MolGAN: An implicit generative model for small molecular graphs,” Sep. 27, 2022, arXiv: arXiv:1805.11973. doi: 10.48550/arXiv.1805.11973.

[4]	C. Vignac, I. Krawczuk, A. Siraudin, B. Wang, V. Cevher, and P. Frossard, “DiGress: Discrete Denoising diffusion for graph generation,” presented at the The Eleventh International Conference on Learning Representations, Sep. 2022.

