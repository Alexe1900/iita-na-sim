import random

import numpy as np
import pandas as pd

import iita_python as iita

class Simulation():
    def __init__(self):
        pass

    @staticmethod
    def generate_surmise_relation(items: int, edge_probability: float=0.3) -> np.ndarray:
        """
        Generates a valid random surmise relation (quasi-order) optimized for KST.
        """
    
        # Step 1: Shuffle and partition items into random clusters
        shuffled = list(items)
        random.shuffle(shuffled)
        k = random.randint(min(3, items), items) # Ensure at least 3 clusters
        clusters = [[] for _ in range(k)]

        # Ensure no empty clusters
        for i in range(k):
            clusters[i].append(shuffled[i])
        for item in shuffled[k:]:
            random.choice(clusters).append(item)
        
        # Step 2: Create compressed adjacency matrix for clusters
        compressed_matrix = np.zeros((k, k), dtype=int)
        
        # Add self-loops on every vertex
        np.fill_diagonal(compressed_matrix, 1)
        
        # Add directed edges from smaller index cluster to larger index cluster
        for i in range(k):
            for j in range(i + 1, k):
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