import numpy as np
from rdkit import Chem
from tensorflow import keras
from utils import smiles_to_graph
from rdkit.Chem.Draw import MolsToGridImage, MolToImage

from model import GraphWGAN

csv_path = keras.utils.get_file(
    "qm9.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
)

if __name__ == "__main__":
    data = []
    with open(csv_path, "r") as f:
        for line in f.readlines()[1:]:
            data.append(line.split(",")[1])

    smiles = data[1000]
    molecule = Chem.MolFromSmiles(smiles)
    MolToImage(molecule).save('mol.png')
    print("SMILES:", smiles)
    print("Num heavy atoms:", molecule.GetNumHeavyAtoms())

    adjacency_tensor, feature_tensor = [], []
    for smiles in data:
        adjacency, features = smiles_to_graph(smiles)
        adjacency_tensor.append(adjacency)
        feature_tensor.append(features)

    wgan = GraphWGAN(discriminator_steps=1)

    wgan.compile(
        optimizer_generator=keras.optimizers.Adam(5e-4),
        optimizer_discriminator=keras.optimizers.Adam(5e-4),
    )
    wgan.fit([np.array(adjacency_tensor, dtype=np.int32), np.array(feature_tensor)], epochs=100, batch_size=64)

    molecules = wgan.sample(batch_size=1000)
    MolsToGridImage(
        [m for m in molecules if m is not None][:25], molsPerRow=5, subImgSize=(150, 150)
    ).save('generated_mol.png')

    with Chem.SDWriter(f'graphs/generated_molecules.sdf') as writer:
        for mol in molecules:
            writer.write(mol)

