from time import perf_counter

import iita_python as iita
import iita_python.additional_ce as iita_add_ce
import numpy as np
import os, json
import pandas as pd

DB_PATH = '../db'

def scan(item_range: range = range(3, 31)):
    done = {}
    errors = {}
    todo = {}

    for items in item_range:
        todo[items] = []
        done[items] = []
        errors[items] = []

        for entry in os.scandir(f"{DB_PATH}/{items}_items"):
            if not entry.is_dir():
                errors[items].append(entry.name)
                continue
            elif not os.path.exists(f"{entry.path}/surmise.csv") or not os.path.exists(f"{entry.path}/knowledge_space.csv") or not os.path.exists(f"{entry.path}/response_nomiss.csv"):
                errors[items].append(entry.name)
                continue
            elif os.path.isdir(f"{entry.path}/iita_output_nomiss/"):
                done[items].append(entry.name)
                continue
            else:
                todo[items].append(entry.name)

    return todo, done, errors

def delete_iita_output_nomiss(item_range: range = range(3, 31)):
    for items in item_range:
        for entry in os.scandir(f"{DB_PATH}/{items}_items"):
            if not entry.is_dir():
                continue
            elif not os.path.exists(f"{entry.path}/surmise.csv") or not os.path.exists(f"{entry.path}/knowledge_space.csv") or not os.path.exists(f"{entry.path}/response_nomiss.csv"):
                continue
            elif os.path.isdir(f"{entry.path}/iita_output_nomiss/"):
                print(f"Deleting {entry.path}/iita_output_nomiss/")
                os.system(f"rm -rf {entry.path}/iita_output_nomiss/")

def run_iita_nomiss(item_range: range = range(3, 31)):
    todo, done, errors = scan(item_range)

    for items in item_range:
        print(f"Running IITA for {items} items")
        print(f"Todo: {len(todo[items])}, Done: {len(done[items])}, Errors: {len(errors[items])}")
        for error in errors[items]:
            print(f"Error: {error}")

        for i_entry, entry in enumerate(sorted(todo[items], key=int)):
            print(f"Entry {entry}, {i_entry+1}/{len(todo[items])}")

            start_time = perf_counter()

            # rp = np.loadtxt(f"{DB_PATH}/{items}_items/{entry}/response_nomiss.csv", delimiter=',', dtype=int)
            rp = pd.read_csv(
                f"{DB_PATH}/{items}_items/{entry}/response_nomiss.csv",
                header=None,
                engine="pyarrow"
            ).to_numpy(dtype=int)

            surmise = pd.read_csv(
                f"{DB_PATH}/{items}_items/{entry}/surmise.csv",
                header=None,
                engine="pyarrow"
            ).to_numpy(dtype=int)

            read_time = perf_counter()

            data = iita.Dataset(rp)
            assert (data.valid_ce_cases.to_numpy() == np.full((data.items, data.items), data.subjects)).all(), "Invalid CE case counts for a complete response pattern"

            rp_time = perf_counter()

            qos = iita.ind_gen(data.ce)

            qo_time = perf_counter()

            os.makedirs(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss", exist_ok=True)

            chosen_qo = {
                "index": None,
                "diff": np.inf,
                "dist": None,
                "noise": None
            }

            closest_qo = {
                "index": None,
                "diff": None,
                "dist": np.inf,
                "noise": None
            }

            for i, qo in enumerate(qos):
                diff, noise = iita.fit_metrics.mini_iita_fit(data, qo)
                dist = np.logical_xor(surmise.flatten(), qo.full_matrix.flatten()).sum()

                if (dist < closest_qo["dist"]):
                    closest_qo["index"] = i
                    closest_qo["diff"] = diff
                    closest_qo["dist"] = int(dist)
                    closest_qo["noise"] = noise

                if (diff < chosen_qo["diff"]):
                    chosen_qo["index"] = i
                    chosen_qo["diff"] = diff
                    chosen_qo["dist"] = int(dist)
                    chosen_qo["noise"] = noise

            qo_process_time = perf_counter()

            with open(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/diff_dist_noise.json", "w") as f:
                json.dump({"chosen": chosen_qo, "closest": closest_qo}, f, indent=4)

            np.savetxt(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/closest_qo.csv", qos[closest_qo["index"]].full_matrix, delimiter=',', fmt='%d')
            np.savetxt(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/chosen_qo.csv", qos[chosen_qo["index"]].full_matrix, delimiter=',', fmt='%d')

            save_time = perf_counter()

            print("Time taken:")
            print(f"Read: {read_time - start_time:.5f}s")
            print(f"RP generation: {rp_time - read_time:.5f}s")
            print(f"QO generation: {qo_time - rp_time:.5f}s")
            print(f"QO processing: {qo_process_time - qo_time:.5f}s")
            print(f"Saving the results: {save_time - qo_process_time:.5f}s")

if __name__ == "__main__":
    # delete_iita_output_nomiss(range(4, 8))
    run_iita_nomiss(range(3, 4))