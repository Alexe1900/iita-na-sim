import random

import numpy as np
import pandas as pd

import iita_python as iita

from blim import BlimSim

from iita_python.quasiorder import QuasiOrder

class Simulation():
    def __init__(self):
        pass

    @staticmethod
    def generate_surmise_relation(items: int, n_clusters: int, edge_probability: float=0.3) -> np.ndarray:
        """
        Generates a valid random surmise relation (quasi-order) optimized for KST.
        """
    
        # Step 1: Shuffle and partition items into random clusters
        shuffled = [i for i in range(items)]
        random.shuffle(shuffled)
        clusters = [[] for _ in range(n_clusters)]

        # Ensure no empty clusters
        for i in range(n_clusters):
            clusters[i].append(shuffled[i])
        for item in shuffled[n_clusters:]:
            random.choice(clusters).append(item)
        
        # Step 2: Create compressed adjacency matrix for clusters
        compressed_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
        
        # Add self-loops on every vertex
        np.fill_diagonal(compressed_matrix, 1)
        
        # Add directed edges from smaller index cluster to larger index cluster
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if random.random() < edge_probability:
                    compressed_matrix[i, j] = 1
        
        # Step 3: Expand the compressed matrix to the full adjacency matrix
        # Create mapping from item to cluster
        item_to_cluster = np.zeros(items, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for item in cluster:
                item_to_cluster[item] = cluster_idx
        
        # Expand compressed matrix to full items×items matrix
        full_matrix = compressed_matrix[item_to_cluster][:, item_to_cluster]
        
        # Step 4: Ensure transitivity by raising matrix to power of items
        # Use boolean matrix to minimize overflow, then convert back to int
        result = np.linalg.matrix_power(full_matrix.astype(bool), items).astype(int)
        
        return result
    
    def generate_from_params(
            self,
            items: int,
            n_clusters: int,
            edge_probability: float,
            subjects: int,
            beta_range: tuple,
            eta_range: tuple,
            missprob_wrong_range: tuple = (0, 0),
            missprob_correct_range: tuple = (0, 0),
            max_noise_sum: float = 0.7,
            sigma: float = 0.1,
            mu: float = 0.5,
            buff: float|np.ndarray = 5,
            seed: int = 42,
            miss: bool = False
        ) -> np.ndarray:
        """
        Generates random response patterns optimized for KST.
        """

        surmise_relation = self.generate_surmise_relation(items, n_clusters, edge_probability)
        qo = iita.QuasiOrder(surmise_relation)

        blim_model = BlimSim(
            beta=np.random.uniform(*beta_range, items),
            eta=np.random.uniform(*eta_range, items),
            qo=qo,
            missprob_wrong=
                np.random.uniform(*missprob_wrong_range, items) if (missprob_wrong_range[1] > 0) else None,
            missprob_correct=
                np.random.uniform(*missprob_correct_range, items) if (missprob_correct_range[1] > 0) else None,
            items=items,
            max_noise_sum=max_noise_sum
        )

        blim_model.set_standard_state_distr(sigma, mu, buff)

        return blim_model.simulate(subjects, miss=miss, seed=seed)
    
if __name__ == "__main__":
    sim = Simulation()
    
    qo = QuasiOrder(sim.generate_surmise_relation(5, 3, 0.5))

    print(qo.full_matrix)
    print(qo.get_knowledge_space())