from scipy.stats.qmc import LatinHypercube

import numpy as np

from iita_python.quasiorder import QuasiOrder

import os, json

from time import sleep

from blim import BlimSim

DB_PATH = '../db'

def scan(item_range: range = range(3, 31)):
    todo = {}
    done = {}
    errors = {}

    for items in item_range:
        todo[items] = []
        done[items] = []
        errors[items] = []

        for entry in os.scandir(f"{DB_PATH}/{items}_items"):
            if not entry.is_dir():
                errors[items].append(entry.name)
                continue
            elif not os.path.exists(f"{entry.path}/surmise.csv") or not os.path.exists(f"{entry.path}/knowledge_space.csv"):
                errors[items].append(entry.name)
                continue
            elif os.path.exists(f"{entry.path}/response_nomiss.csv"):
                done[items].append(entry.name)
                continue
            else:
                todo[items].append(entry.name)

    return todo, done, errors

def generate_params(n):
    # eta lower bound, upper bound
    # beta lower bound, upper bound

    # knowledge state probability distribution sigma, mu, flatness
    # simulation of knowledge state probability distribution: desmos.com/calculator/hi0i6doblp

    # subject count (int)

    lhs = LatinHypercube(d=8)

    sample = lhs.random(n)

    sample[:, 0] = sample[:, 0] # beta lower bound
    sample[:, 1] = sample[:, 1] # beta upper bound
    sample[:, 2] = sample[:, 2] # eta lower bound
    sample[:, 3] = sample[:, 3] # eta upper bound
    sample[:, 4] = sample[:, 4] * (0.7 - 0.3) + 0.3 # sigma
    sample[:, 5] = sample[:, 5] * (0.8 - 0.2) + 0.2 # mu
    sample[:, 6] = sample[:, 6] # flatness
    sample[:, 7] = np.round(sample[:, 7] * (10000 - 10) + 10) # subject count

    # ensuring lower bound < upper bound for beta and eta
    mask_eta = sample[:, 0] > sample[:, 1]
    sample[mask_eta, 0], sample[mask_eta, 1] = sample[mask_eta, 1].copy(), sample[mask_eta, 0].copy()

    mask_beta = sample[:, 2] > sample[:, 3]
    sample[mask_beta, 2], sample[mask_beta, 3] = sample[mask_beta, 3].copy(), sample[mask_beta, 2].copy()

    np.random.shuffle(sample)

    return sample

def params_to_dict(params):
    return {
        'beta_range': [params[0], params[1]],
        'eta_range': [params[2], params[3]],
        'sigma': params[4],
        'mu': params[5],
        'flatness': params[6],
        'subjects': params[7]
    }

def generate_resp_nomiss(item_range: range = range(3, 31)):
    todo, done, errors = scan(item_range)

    for items in item_range:
        print(f"Generating responses for {items} items...")

        print(f"Todo: {len(todo[items])}, Done: {len(done[items])}, Errors: {len(errors[items])}")

        if (len(errors[items]) > 0 and len(errors[items]) < 10):
            print(f"Errors: {errors[items]}")
        elif (len(errors[items]) >= 10):
            print(f"Errors: {len(errors[items])} errors, too many to display.")
            break

        params = generate_params(len(todo[items]))

        for i, entry in enumerate(sorted(todo[items], key=int)):
            print(f"Generating responses for {items} items, entry {entry} ({i + 1}/{len(todo[items])})...")
            with open(f"{DB_PATH}/{items}_items/{entry}/data.json", 'r') as f:
                metadata = json.load(f)

            surmise_relation = np.loadtxt(f"{DB_PATH}/{items}_items/{entry}/surmise.csv", delimiter=',', dtype=int)

            try:
                knowledge_space = np.loadtxt(f"{DB_PATH}/{items}_items/{entry}/knowledge_space.csv", delimiter=',', dtype=int)
                qo = QuasiOrder(surmise_relation, knowledge_space)
            except ValueError:
                qo = QuasiOrder(surmise_relation)
                knowledge_space = qo.get_knowledge_space().astype(int)
                np.savetxt(f"{DB_PATH}/{items}_items/{entry}/knowledge_space.csv", knowledge_space, delimiter=",", fmt="%d")

            blim_model = BlimSim(
                beta=np.random.uniform(params[i, 0], params[i, 1], items),
                eta=np.random.uniform(params[i, 2], params[i, 3], items),
                qo=qo,
                items=items,
                max_noise_sum=1
            )

            blim_model.set_standard_state_distr(params[i, 4], params[i, 5], params[i, 6])

            resp_nomiss = blim_model.simulate(subjects=int(params[i, 7]), miss=False)

            np.savetxt(f"{DB_PATH}/{items}_items/{entry}/response_nomiss.csv", resp_nomiss, delimiter=',', fmt='%d')

            with open(f"{DB_PATH}/{items}_items/{entry}/metadata_rp_nomiss.json", 'w') as f:
                json.dump({
                    **metadata,
                    **params_to_dict(params[i]),
                }, f, indent=4)

            with open(f"{DB_PATH}/{items}_items/{entry}/beta_eta.json", 'w') as f:
                json.dump({
                    "beta": blim_model.beta.tolist(),
                    "eta": blim_model.eta.tolist(),
                    "states_distr": blim_model.states_distr.tolist(),
                }, f, indent=4)

if __name__ == "__main__":
    def tr():
        try:
            generate_resp_nomiss(range(30, 31))
        except TimeoutError:
            print("FAIL")
            os.system('open -a OneDrive')
            sleep(10)
            tr()
    tr()