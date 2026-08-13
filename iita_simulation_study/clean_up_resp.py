import os

DB_PATH = '../../db'

def delete_responses(item_range: range = range(3, 31)):
    for items in item_range:
        for entry in os.scandir(f"{DB_PATH}/{items}_items"):
            if not entry.is_dir():
                continue

            if os.path.exists(f"{entry.path}/response_miss.csv"):
                os.remove(f"{entry.path}/response_miss.csv")
            if os.path.exists(f"{entry.path}/response_nomiss.csv"):
                os.remove(f"{entry.path}/response_nomiss.csv")
            if os.path.exists(f"{entry.path}/full_metadata.json"):
                os.remove(f"{entry.path}/full_metadata.json")
            if os.path.exists(f"{entry.path}/beta_eta_missprob.json"):
                os.remove(f"{entry.path}/beta_eta_missprob.json")

delete_responses(range(3, 4))