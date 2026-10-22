from scipy.stats.qmc import LatinHypercube

import numpy as np

from iita_python.quasiorder import QuasiOrder

import os, json

from blim import BlimSim

DB_PATH = '../db'

def delete_responses(item_range: range = range(3, 31)):
    for items in item_range:
        for entry in os.scandir(f"{DB_PATH}/{items}_items"):
            if not entry.is_dir():
                continue

            if os.path.exists(f"{entry.path}/response_mcar.csv"):
                os.remove(f"{entry.path}/response_mcar.csv")
            if os.path.exists(f"{entry.path}/metadata_rp_mcar.json"):
                os.remove(f"{entry.path}/metadata_rp_mcar.json")

def scan(item_range: range = range(3, 31)):
    todo_mcar = {}
    done = {}
    errors = {}

    for items in item_range:
        todo_mcar[items] = []
        done[items] = []
        errors[items] = []

        for entry in os.scandir(f"{DB_PATH}/{items}_items"):
            if not entry.is_dir():
                errors[items].append(entry.name)
                continue
            elif not os.path.exists(f"{entry.path}/surmise.csv") or not os.path.exists(f"{entry.path}/knowledge_space.csv"):
                errors[items].append(entry.name)
                continue
            elif os.path.exists(f"{entry.path}/response_mcar.csv"):
                done[items].append(entry.name)
                continue
            else:
                todo_mcar[items].append(entry.name)

    return todo_mcar, done, errors

def generate_params(n):
    # missingness probability lower bound, upper bound

    lhs = LatinHypercube(d=2)

    sample = lhs.random(n)

    mask = sample[:, 0] > sample[:, 1]
    sample[mask, 0], sample[mask, 1] = sample[mask, 1], sample[mask, 0]

    np.random.shuffle(sample)
    return sample

def generate_resp_mcar(item_range: range = range(3, 31)):
    todo_mcar, done, errors = scan(item_range)

    for items in item_range:
        print(f"Generating responses for {items} items...")
        print(f"Done: {len(done[items])}, To do: {len(todo_mcar[items])}, Errors: {len(errors[items])}")

        for error in errors[items]:
            print(f"Error in {items} items, entry {error}")

        params = generate_params(len(todo_mcar[items]))

        for i, entry in enumerate(sorted(todo_mcar[items], key=int)):
            print(f"Generating responses for {items} items, entry {entry} ({i + 1}/{len(todo_mcar[items])})...")
            with open(f"{DB_PATH}/{items}_items/{entry}/beta_eta.json", "r") as f:
                metadata = json.load(f)
                beta = np.array(metadata["beta"])
                eta = np.array(metadata["eta"])
                states_distr = np.array(metadata["states_distr"])

            with open(f"{DB_PATH}/{items}_items/{entry}/metadata_rp_nomiss.json", "r") as f:
                metadata = json.load(f)
                subjects = metadata["subjects"]
                beta_range = metadata["beta_range"]
                eta_range = metadata["eta_range"]
                sigma = metadata["sigma"]
                mu = metadata["mu"]
                flatness = metadata["flatness"]

            with open(f"{DB_PATH}/{items}_items/{entry}/knowledge_space.csv", "r") as f:
                states = np.loadtxt(f, delimiter=",")

            with open(f"{DB_PATH}/{items}_items/{entry}/surmise.csv", "r") as f:
                surmise_relation = np.loadtxt(f, delimiter=",", dtype=int)

            qo = QuasiOrder(surmise_relation, states)

            blim_model = BlimSim(
                beta=beta,
                eta=eta,
                qo=qo,
                items=items,
                max_noise_sum=1,
                missprob_wrong=np.random.uniform(params[i, 0], params[i, 1], items),
                missprob_correct=np.random.uniform(params[i, 0], params[i, 1], items)
            )

            blim_model.set_standard_state_distr(sigma, mu, flatness)

            resp_mcar = blim_model.simulate(subjects=subjects, miss=True)
            resp_mcar = np.where(np.isnan(resp_mcar), -1, resp_mcar).astype(int)

            np.savetxt(f"{DB_PATH}/{items}_items/{entry}/response_mcar.csv", resp_mcar, delimiter=',', fmt='%d')

            with open(f"{DB_PATH}/{items}_items/{entry}/metadata_rp_mcar.json", 'w') as f:
                json.dump({
                    "beta_range": beta_range,
                    "eta_range": eta_range,
                    "missprob_range": [params[i, 0], params[i, 1]],
                    "sigma": sigma,
                    "mu": mu,
                    "flatness": flatness,
                    "subjects": subjects,
                    "missprob": blim_model.missprob_wrong.tolist(),
                    "total_missingness": np.sum(resp_mcar == -1) / (resp_mcar.shape[0] * resp_mcar.shape[1])
                }, f, indent=4)

if __name__ == "__main__":
    def tr():
        try:
            generate_resp_mcar(range(30, 31))
        except TimeoutError:
            os.system('open -a OneDrive')
            tr()
    tr()