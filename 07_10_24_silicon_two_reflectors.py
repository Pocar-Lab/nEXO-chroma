#!/usr/bin/env python
from PocarChroma.geometry_manager import geometry_manager
from PocarChroma.run_manager import run_manager
import time


"""
Example showing how to run a simulation with a script file rather than from the command line.

Here perhaps include description of what the simulation is testing for archival reasons.

Run this file from within the Chroma container with `python ./ExampleAnalysis.py`
"""


def main():

    experiment_name = "SiliconeFlippedSourceLower"
    #experiment_name = "FlippedSourceTall"
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = True
    plots = []
    # nums =  [9, 10, 11, 12, 13, 14, 15, 16] # tall
    nums = [2, 3, 4, 5, 6, 7, 8, 9] #lower

    # pairs = [[10, 11], [9, 12], [13, 15], [14, 16]] # tall
    pairs = [[2, 5], [3, 9], [6, 4], [7, 8]] # lower
    labels = [
        "silicon-lower-2-5",
        "silicon-lower-3-9",
        "silicon-lower-6-4",
        "silicon-lower-7-8",
    ]

    #exclusions = [[f"reflector{i}" for i in excl] for excl in exclusions]
    ptes = []
    ptes_err = []
    for pair in pairs:
        p = set(pair)
        n = set(nums)
        exclusion = [f"reflector{i}" for i in n - p]
        print(f"Experiment Name:       {experiment_name }")
        print(f"Number of particles:   {str(num_particles) }")
        print(f"Random seed:           {str(seed) }")
        print(f"Run ID:                {str(run_id) }")
        print(f"Visualize:             {str(visualize) }")
        print(f"Plots:                 {plots}")
        print(f"Excluded reflectors:   {exclusion}")

        gm = geometry_manager(
            experiment_name=experiment_name,
            run_id=run_id,
            visualize=visualize,
            exclude=exclusion,
        )
        rm = run_manager(
            geometry_manager=gm,
            experiment_name=experiment_name,
            random_seed=seed,
            num_particles=num_particles,
            run_id=run_id,
            plots=plots,
            write=False,
        )
        pte = rm.ana_man.photon_transmission_efficiency
        ptes.append(pte)
        ptes_err.append(rm.ana_man.pte_st_dev)
        run_id += 1
    print("-----------------------------")
    print("Conclusion")
    for i in range(len(labels)):
        print(f"Run{i+1}: {labels[i]: <35} - {ptes[i]} +- {ptes_err[i]}")
    print(labels)
    print(ptes)
    print(ptes_err)


if __name__ == "__main__":
    s = time.time()
    main()
    e = time.time()
    print(f"The simulation run time is: {e - s} s")
