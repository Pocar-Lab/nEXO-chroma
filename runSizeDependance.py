import subprocess
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_simulation(r0):
    result = subprocess.run(
        ["python", "sizedependance.py", str(r0), str(1)],
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["python", "sizedependance.py", str(r0), str(0)], capture_output=True, text=True
    )
    return result.stdout


if __name__ == "__main__":
    start_time = time.time()
    print("Running simulation with 2 threads")
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(run_simulation, r) for r in np.linspace(0.001, 5, 30)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            out = future.result()
            # print(out)

    end_time = time.time()
    print(f"Total run time for all simulations: {end_time - start_time} s")
