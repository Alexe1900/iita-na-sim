import numpy as np
import pandas as pd
import matlab.engine as mlb_eng

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

    mlb_runner = mlb_eng.start_matlab()

    items: int

    rng: np.random.Generator = np.random.default_rng()

    def __init__(
            self,
            beta: np.ndarray,
            eta: np.ndarray,
            qo: QuasiOrder,
            missprob_wrong: np.ndarray | None,
            missprob_correct: np.ndarray | None,
            items: int,
            max_noise_sum: float = 1,
            seed: int = 42
        ):
        self.beta = beta
        self.eta = eta
        self.qo = qo
        self.states = qo.get_knowledge_space()
        self.states_n = self.states.shape[0]
        self.states_distr = np.ones(self.states_n) / self.states_n
        self.missprob_wrong = missprob_wrong
        self.missprob_correct = missprob_correct
        self.items = items

        self.set_seed(seed)

        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.mlb_runner.addpath(script_dir, nargout=0)

        self.sort_beta_eta(max_sum=max_noise_sum)

    def sort_beta_eta(self, max_sum: float = 1):
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

    def set_standard_state_distr(self, sigma: float, mu: float = 0.5, mult: float|np.ndarray = 1, buff: float|np.ndarray = 0):
        def gauss(x):
            x_norm = x / self.items
            return np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        
        self.states_distr = gauss(self.states.sum(axis=1))
        self.states_distr *= mult
        self.states_distr += buff
        self.states_distr /= self.states_distr.sum()
    
    def simulate(self, subjects: int, seed: int, miss: bool = False) -> np.ndarray:
        self.set_seed(seed)

        if (not miss):
            _, _, samp = self.mlb_runner.sampleBLIM({
                'beta': self.beta,
                'eta': self.eta,
                'states': self.states,
                'pi': self.states_distr,
            }, np.double(subjects), nargout=3)
            return np.array(samp)
        else:
            sample = self.mlb_runner.samplenan({
                'beta': self.beta.astype(np.double),
                'eta': self.eta.astype(np.double),
                'states': self.states.astype(np.double),
                'pi': self.states_distr.astype(np.double),
                'mu0': self.missprob_wrong.astype(np.double),
                'mu1': self.missprob_correct.astype(np.double)
            }, np.double(subjects), nargout=1)
            return np.array(sample['data'])

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.mlb_runner.rng(seed, nargout=0)

if __name__ == "__main__":
    beta = np.array([0.1] * 5)[:, np.newaxis]
    eta = np.array([0.1] * 5)[:, np.newaxis]

    missprob_wrong = np.array([0] * 5)[:, np.newaxis]
    missprob_correct = np.array([0.5] * 5)[:, np.newaxis]

    surm = np.array([
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
    ])

    blim_sim = BlimSim(
        beta,
        eta,
        iita.quasiorder.QuasiOrder(surm),
        missprob_wrong,
        missprob_correct,
        items=5,
        max_noise_sum=0.7,
        seed=41
    )

    blim_sim.set_standard_state_distr(sigma=0.4, mu=0.5, mult=1, buff=0)

    # print(blim_sim.states)

    # print(blim_sim.states_distr)

    samp = blim_sim.simulate(20, 41, miss=True)
    print(samp)