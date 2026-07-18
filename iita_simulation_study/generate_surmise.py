from main import Simulation
from scipy.stats.qmc import LatinHypercube

import numpy as np

from iita_python.quasiorder import QuasiOrder

import os, json

def generate(n: int):
    # Choosing parameters

    # items: 3 - 25
        # chosen between 3 and 26, then rounded down
    # edge probaability: 0.1 - 0.9
    # number of clusters chosen later
    
    lhs = LatinHypercube(d=2)

    sample = lhs.random(n)

    sample[:, 0] = np.floor(sample[:, 0] * (26 - 3) + 3)
    sample[:, 1] = sample[:, 1] * (0.9 - 0.1) + 0.1

    for i in range(n):
        items = int(sample[i, 0])
        edge_probability = sample[i, 1]

        n_clusters = np.random.randint(2, items+1)  # Randomly choose number of clusters between 2 and number of items

        surm = Simulation.generate_surmise_relation(items, n_clusters=n_clusters, edge_probability=edge_probability).astype(int)
        qo = QuasiOrder(surm)

        interconnectivity = (np.sum(surm) - items) / (items * items - items)  # Exclude diagonal

        data = {
            "items": items,
            "n_clusters": n_clusters,
            "edge_probability": edge_probability,
            "interconnectivity": interconnectivity
        }

        ks = qo.get_knowledge_space().astype(int)

        # save data

        os.makedirs(f"./db/{items}_items/{i}", exist_ok=True)
        np.savetxt(f"./db/{items}_items/{i}/surmise.csv", surm, delimiter=",", fmt="%d")
        np.savetxt(f"./db/{items}_items/{i}/knowledge_space.csv", ks, delimiter=",", fmt="%d")
        with open(f"./db/{items}_items/{i}/data.json", "w") as f:
            json.dump(data, f)
        
        print(f'Generated sample {i}')

if __name__ == "__main__":
    generate(1000)