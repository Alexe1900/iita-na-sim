import numpy as np
import pandas as pd
import oct2py as o2p

import iita_python as iita
from iita_python.quasiorder import QuasiOrder

class BlimSim():
    beta: np.ndarray
    eta: np.ndarray

    qo: QuasiOrder

    states: np.ndarray
    states_n: int
    states_distr: np.ndarray

    miss: np.ndarray | None

    o2p_runner: o2p.Oct2Py

    items: int

    def __init__(
            self,
            beta: np.ndarray,
            eta: np.ndarray,
            qo: QuasiOrder,
            miss: np.ndarray | None,
            items: int,
            max_noise_sum: float = 0.7
        ):
        self.o2p_runner = o2p.Oct2Py()
        self.beta = beta
        self.eta = eta
        self.qo = qo
        self.states = qo.get_knowledge_space()
        self.states_n = states.shape[0]
        self.states_distr = np.ones(self.states_n) / self.states_n
        self.miss = miss
        self.items = items

        self.o2p_runner.octave.addpath('./sample.m')

        self.sort_beta_eta(max_sum=max_noise_sum)

    def sort_beta_eta(self, max_sum: float = 0.7):
        predecessors_count = self.qo.full_matrix.sum(axis=0)

        sorting_order = np.argsort(predecessors_count)

        beta_sorted = self.beta[sorting_order]
        eta_sorted = self.eta[np.flip(sorting_order)]

        summed = beta_sorted + eta_sorted
        if summed.max() > max_sum:
            beta_sorted = beta_sorted / summed.max() * max_sum
            eta_sorted = eta_sorted / summed.max() * max_sum
        
        self.beta = beta_sorted
        self.eta = eta_sorted

    def set_standard_state_distr(self, sigma: float, mu: float = 0.5, buff: float|np.ndarray = 0):
        def gauss(x):
            x_norm = x / self.items
            return np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        
        self.states_distr = gauss(self.states.sum(axis=0))
        self.states_distr += buff
        self.states_distr /= self.states_distr.sum()
    
    def simulate(self, subjects: int, miss: bool = True):
        pat, freq, samp = self.o2p_runner.sample({
            'beta': self.beta,
            'eta': self.eta,
            'states': self.states,
            'pi': self.states_distr,
            'miss': self.miss if miss else None
        }, subjects)

        return samp

if __name__ == "__main__":
    beta = np.array([0] * 7)
    eta = np.array([0] * 7)

    states = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    blim_sim = BlimSim(beta, eta, states, miss=None, items=7)

    blim_sim.set_standard_state_distr(sigma=0.5, mu=0.5, buff=0)

    print(blim_sim.states_distr)

    # samp = blim_sim.simulate(20, miss=False)
    # print(samp)