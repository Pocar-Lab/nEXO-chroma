import subprocess
import time
import numpy as np
from tqdm import tqdm


def run_simulation(r0):
    result = subprocess.run(
        ["python", "sizedependance.py", str(r0)], capture_output=True, text=True
    )
    return result.stdout


if __name__ == "__main__":
    start_time = time.time()
    for r in tqdm(np.linspace(0.01, 4, 25)):
        # print(f"Running simulation with r0 = {r}")
        out = run_simulation(r)
        # print(out)
    end_time = time.time()
    print(f"Total run time for all simulations: {end_time - start_time} s")
