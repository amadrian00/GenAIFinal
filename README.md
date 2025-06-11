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
│   ├── __init__.py
│   ├── ConditionalCGVAE-master     # CCGVAE model
│   │   ├── data
│   │   │   └── make_dataset.py     # File for constructing the dataset.
│   │   ├── CCGVAE.py               # Main script of the CCGVAE model.
│   │   └── setup.bash              # Bash for automatic creation of the environment.  
│   ├── DiGress-main                # DiGress model
│   │   └── src
│   │      └── main.py              # Main script of the CCGVAE model.
│   ├── graph-generation-master     # GraphRNN model
│   │   ├── environment.yml         # Environment configuration.
│   │   └── main.py                 # Main script of the CCGVAE model.
│   └── MolGAN                      # MolGAN model
│       ├── environment.yml         # Environment configuration.
│       └── main.py                 # Main script of the MolGAN model.
│
├── main.py                         # Main script to perform evaluation of the models.
├── metrics.py                      # Definition of the validity, uniqueness and novelty metrics.
└── README.md                       # Project documentation
```

---

## Code

In the project structure section, just the relevant parts were included. 
Since the code incorporates code from four different projects, there are many files that are not relevant for this project.

The compiled models are from different repositories, all of them created in different years with different library versions.
This makes it impossible to join all of them under one environment in a reasonable amount of time.
Therefore, each of the reviewed models includes its own environment and libraries.
The code for training, fine-tuning and sampling should be undergone under the correspondant environment for each model.
Special caution must be taken when setting the environments for each model since even the slightest change may impede the correct functioning of the process.

The changes made to the original code are minimal and limited to adapting functionality to non-available libraries or correcting minor mistakes.

Knowing the great complexity that executing the models may take and given that anyway the comparison tests should be executed on a separate environment, the sampled molecules for each model have been saved.
This way executing the `main.py` is enough for running the metric calculation.


---

## Usage
- Running `main.py` is enough to run all the training and the test battery.

### GraphRNN

```bash
    CUDA_VISIBLE_DEVICES=6 python3 main.py
```

### CCGVAE
- Excuting the `setup.bash` will set-up the environment.
- Then the execution of the model is (generation is 0 for training, 1 for generating, 2 for reconstructing):
```bash
    CUDA_VISIBLE_DEVICES=1 python CCGVAE.py --dataset qm9 --config '{"generation":0, "log_dir":"./results", "use_mask":false}'
```

### MolGAN

```bash
    CUDA_VISIBLE_DEVICES=7 python3 main.py 
```

### DiGress

- The creation of the environment is a bit more complex, so I refer to the original repository where it is explained in detail (https://github.com/cvignac/DiGress/tree/main).
- Then executing as:
```bash
    CUDA_VISIBLE_DEVICES=1 python3 main.py dataset=qm9
```


## References
[1]	J. You, R. Ying, X. Ren, W. Hamilton, and J. Leskovec, “GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models,” in Proceedings of the 35th International Conference on Machine Learning, PMLR, Jul. 2018, pp. 5708–5717.

[2]	D. Rigoni, N. Navarin, and A. Sperduti, “Conditional constrained graph variational autoencoders for molecule design,” in 2020 IEEE symposium series on computational intelligence (SSCI), IEEE, 2020, pp. 729–736.

[3]	N. D. Cao and T. Kipf, “MolGAN: An implicit generative model for small molecular graphs,” Sep. 27, 2022, arXiv: arXiv:1805.11973. doi: 10.48550/arXiv.1805.11973.

[4]	C. Vignac, I. Krawczuk, A. Siraudin, B. Wang, V. Cevher, and P. Frossard, “DiGress: Discrete Denoising diffusion for graph generation,” presented at the The Eleventh International Conference on Learning Representations, Sep. 2022.

## Contact
amadrian@korea.ac.kr