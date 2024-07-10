import os
import pandas as pd

# NAME = "SiliconeFlippedSource"
projects = [
    name
    for name in os.listdir("data_files/data/")
    if os.path.isdir(f"data_files/data/{name}")
]

print("STATUS     - PROJECT NAME")
print("--------------------------------")

for NAME in projects:
    try:
        df = pd.read_csv(f"data_files/data/{NAME}/geometry_components_{NAME}.csv")
    except FileNotFoundError:
        print(f"NoGeometry - {NAME}")
        continue
    not_found = []
    for location in df["stl_filepath"]:
        loc = location.strip("/workspace/")
        isFound = os.path.isfile(loc)
        # print(f"{loc} - {isFound}")
        if not isFound:
            not_found.append(loc)
    # print(f"{len(not_found)} files not found as expected.")
    if len(not_found) == 0:
        print(f"OK         - {NAME}")
    else:
        print(f"ERR        - {NAME}")
        print(not_found)
