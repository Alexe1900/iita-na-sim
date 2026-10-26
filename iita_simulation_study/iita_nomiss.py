import iita_python as iita
import iita_python.additional_ce as iita_add_ce
import numpy as np
import os, json

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

def run_iita_nomiss(item_range: range = range(3, 31)):
    todo, done, errors = scan(item_range)

    for items in item_range:
        print(f"Running IITA for {items} items")
        print(f"Todo: {len(todo[items])}, Done: {len(done[items])}, Errors: {len(errors[items])}")
        for error in errors[items]:
            print(f"Error: {error}")

        for i_entry, entry in enumerate(sorted(todo[items], key=int)):
            print(f"Entry {entry}, {i_entry+1}/{len(todo[items])}")

            rp = np.loadtxt(f"{DB_PATH}/{items}_items/{entry}/response_nomiss.csv", delimiter=',', dtype=int)

            data = iita.Dataset(rp)
            ce_relative = np.where(data.valid_ce_cases != 0, data.ce / data.valid_ce_cases, 0.5) # if items not observed together, set to 0.5
            ce_item_mean_imp = iita_add_ce.missing_value_substitution_ce(data.rp)

            qos_absolute = iita.ind_gen(data.ce)
            qos_relative = iita.ind_gen(ce_relative)
            qos_item_mean_imp = iita.ind_gen(ce_item_mean_imp)

            os.makedirs(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/qos_absolute", exist_ok=True)
            os.makedirs(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/qos_relative", exist_ok=True)
            os.makedirs(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/qos_item_mean_imp", exist_ok=True)

            for type_str, qos in zip(["absolute", "relative", "item_mean_imp"], [qos_absolute, qos_relative, qos_item_mean_imp]):
                noise_diff = {
                    "noise": [],
                    "diff": [],
                    "best_qo": 0,
                    "best_diff": float("inf"),
                }
                for i, qo in enumerate(qos):
                    diff, noise = iita.fit_metrics.mini_iita_fit(data, qo)
                    noise_diff["diff"].append(diff)
                    noise_diff["noise"].append(noise)

                    np.savetxt(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/qos_{type_str}/quasiorder_{i}.csv", qo.full_matrix, delimiter=',', fmt='%d')

                    if diff < noise_diff["best_diff"]:
                        noise_diff["best_diff"] = diff
                        noise_diff["best_qo"] = i

                with open(f"{DB_PATH}/{items}_items/{entry}/iita_output_nomiss/qos_{type_str}/noise_diff.json", "w") as f:
                    json.dump(noise_diff, f, indent=4)

if __name__ == "__main__":
    run_iita_nomiss(range(3, 10))