# Evaluating the Evolution of Graph Generative Models: Autoregressive, VAE, GAN and Diffusion Approaches

 

## Overview

This project compares different generative artificial intelligence methods for graphs, applied specifically to the case of molecule generation.
The model's code is almost identical to the original in its respective repository, with only minimal changes made.
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
│   ├── ConditionalCGVAE-master             # CCGVAE model
│   │   ├── data
│   │   │   ├── make_dataset.py             # File for constructing the dataset.
│   │   │   └── qm9.smi                     # QM9 dataset in SMILES form.
│   │   ├── CCGVAE.py                       # Main script of the CCGVAE model.
│   │   └── setup.bash                      # Bash for automatic creation of the environment.  
│   ├── DiGress-main                        # DiGress model
│   │   ├── data                            # Folder where qm9 dataset is stored.
│   │   ├── outputs                         # Folder where output is saved for every training run.
│   │   └── src
│   │      └── main.py                      # Main script of the CCGVAE model.
│   ├── graph-generation-master             # GraphRNN model
│   │   ├── graphs                          # Generated graphs folder
│   │   ├── environment.yml                 # Environment configuration.
│   │   └── main.py                         # Main script of the CCGVAE model.
│   └── MolGAN                              # MolGAN model
│       ├── graphs
│       │   └── generated_molecules.sdf     # Saved graphs
│       ├── environment.yml                 # Environment configuration.
│       └── main.py                         # Main script of the MolGAN model.
│
├── main.py                                 # Main script to perform evaluation of the models.
├── metrics.py                              # Definition of the validity, uniqueness and novelty metrics.
└── README.md                               # Project documentation
```

---

## Code

In the project structure section, only the relevant parts were included, there are some other folders and files which are do not need any indication for the scope of this project. 
Since the code incorporates components from four different projects, many files are not relevant for this project.

The compiled models come from different repositories, all created in different years with different library versions. 
This makes it impossible to unify them under a single environment in a reasonable amount of time. Therefore, each of the reviewed models includes its own environment and libraries. 
The code for training, fine-tuning, and sampling must be executed within the corresponding environment for each model. 
Special caution must be taken when setting up the environments for each model, since even the slightest change may impede the correct functioning of the process.

The changes made to the original code are minimal and limited to adapting functionality to unavailable libraries or correcting minor mistakes. 
the changes are minor, the task of understanding, adapting, and getting the code to work from such a diverse set of models is not trivial. 
The environment of CCGVAE is especially sensitive to library changes; if any error occurs during execution, it means that some dependency is not correctly installed. 
This is why a setup script is provided for this model; nevertheless, depending on the local machine, some unexpected errors may arise.

Knowing the great complexity involved in executing the models, and given that the comparison tests should be run in separate environments anyway, the sampled molecules for each model have been saved.
This way, running `main.py` is enough for running the metric calculation.


---

## Usage
- In an environment with `Python 3.9`, installing the libraries listed in `requirements.txt` is enough to set up the main environment.
- Running `main.py` is enough to run all the training and the test battery.

### GraphRNN
- Creating an environment using the `environment.yml` will set up the environment.
- Then execute:
```bash
CUDA_VISIBLE_DEVICES=6 python3 main.py
```

### CCGVAE
- Excuting the `setup.bash` will set up the environment.
- Then, the model can be run with (generation values: 0 for training, 1 for generating, 2 for reconstructing):
```bash
CUDA_VISIBLE_DEVICES=1 python CCGVAE.py --dataset qm9 --config '{"generation":0, "log_dir":"./results", "use_mask":false}'
```

- Training is quite costly and time-consuming; therefore, results are presented for a minimally trained model.

### MolGAN
- Creating an environment using the `environment.yml` will set up the environment.
- Then execute:
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