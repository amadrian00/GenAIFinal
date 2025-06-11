#!/usr/bin/env bash

s1=$1
s2="install"
s3="remove"
s4="pretrained"

if [[ $s1 == $s2 ]]; then
    echo "-------------------------------------------------------------------------------------------------------------"
    echo "---------------------------------------------------  DATA - -------------------------------------------------"
    echo "-------------------------------------------------------------------------------------------------------------"
    cd data
    wget --no-check-certificate -O "qm9.zip" https://www.dropbox.com/s/cfra3r50j89863x/qm9.zip?dl=0
    unzip -a qm9.zip
    rm qm9.zip

    wget --no-check-certificate -O "zinc.zip" https://www.dropbox.com/s/rrisjasazovyouf/zinc.zip?dl=0
    unzip -a zinc.zip
    rm zinc.zip


    echo "-------------------------------------------------------------------------------------------------------------"
    echo "---------------------------------------------------  UTILS - ------------------------------------------------"
    echo "-------------------------------------------------------------------------------------------------------------"
    cd ../utils
    wget --no-check-certificate https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
    wget --no-check-certificate https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz

echo "------------------------------------------------------------------------------------------------------------"
    echo "---------------------------------------------------  CCGVVAE  -----------------------------------------------"
    echo "------------------------------------------------------------------------------------------------------------"

    CONDA_BASE=$(conda info --base)
    if [ -z "$CONDA_BASE" ]; then
        echo "Error: conda not found or not initialized. Please ensure conda is in your PATH and run 'conda init' manually."
        exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if conda env list | grep -q "ccgvae"; then
        echo "Conda environment 'ccgvae' already exists. Removing it for a clean installation..."
        conda env remove -n ccgvae --y
    fi

    conda env create -f ccgvae_env.yml --name ccgvae

    conda activate ccgvae

    pip install --upgrade pip

    pip install Cython==0.29.1

    pip install -r ccgvae_env_requirements.txt

    conda deactivate

elif [[ $s1 == $s3 ]]; then
    CONDA_BASE=$(conda info --base)
    if [ -z "$CONDA_BASE" ]; then
        echo "Error: conda not found or not initialized. Please ensure conda is in your PATH and run 'conda init' manually."
        exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    conda deactivate
    conda env remove -n ccgvae --all

elif [[ $s1 == $s4 ]]; then
    echo "To be implemented."

else
    echo 'Uso: install | remove | pretrained'
fi
