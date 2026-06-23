import numpy as np
import pandas as pd
import iita_python as iita
from iita_python.quasiorder import QuasiOrder

def generate_rp_simple(quasi_order: QuasiOrder, subjects: int, careless: float, lucky: float) -> np.ndarray:
    """
    Generate synthetic response patterns based on a given quasi-order, subject count,
    careless error and lucky guess rates.
    """

    knowledge_space = quasi_order.get_knowledge_space()

    rng = np.random.default_rng()
    ks_indices = rng.choice(knowledge_space.shape[0], size=subjects, replace=True)
    data = knowledge_space[ks_indices]

    correct_indices = np.argwhere(data == 1)
    careless_indices = rng.choice(len(correct_indices), size=int(careless * len(correct_indices)), replace=False)
    rows, cols = correct_indices[careless_indices].T
    data[rows, cols] = 0

    wrong_indices = np.argwhere(data == 0)
    lucky_indices = rng.choice(len(wrong_indices), size=int(lucky * len(wrong_indices)), replace=False)
    rows, cols = wrong_indices[lucky_indices].T
    data[rows, cols] = 1

    return data

def generate_qo_simple(items: int, interconnectivity_ratio: float) -> QuasiOrder:
    """
    Generate a random quasi-order for a specified number of items and a given interconnectivity ratio.
    The interconnectivity ratio determines the density of the quasi-order's relation matrix.
    """

    matrix = np.eye(items)

    to_add = int(interconnectivity_ratio * (items * (items - 1)))

    # randomly add interconnectivity relations
    zero_indices = np.argwhere(matrix == 0)
    chosen = np.random.default_rng().choice(len(zero_indices), size=to_add, replace=False)
    rows, cols = zero_indices[chosen].T
    matrix[rows, cols] = 1

    # ensure transitivity
    matrix = np.linalg.matrix_power(matrix.astype(bool), items)

    return QuasiOrder(matrix.astype(int))