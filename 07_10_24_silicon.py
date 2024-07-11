#!/usr/bin/env python
from analysis.geometry_manager import geometry_manager
from analysis.run_manager import run_manager
import time


"""
Example showing how to run a simulation with a script file rather than from the command line.

Here perhaps include description of what the simulation is testing for archival reasons.

Run this file from within the Chroma container with `python ./ExampleAnalysis.py`
"""


def main():

    experiment_name = "SiliconeFlippedSourceLower"
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = True
    plots = []

    exclusions = [
        [2, 3, 4, 5, 6, 7, 8, 9],
        [],
        [2, 3, 4, 5],
        [
            6,
            7,
            8,
            9,
        ],
    ]

    labels = [
        "silicon-lower-noreflector",
        "silicon-lower-8reflector",
        "silicon-lower-4reflector-outer",
        "silicon-lower-4reflector-inner",
    ]

    exclusions = [[f"reflector{i}" for i in excl] for excl in exclusions]
    ptes = []
    ptes_err = []
    for exclusion in exclusions:
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


if __name__ == "__main__":
    s = time.time()
    main()
    e = time.time()
    print(f"The simulation run time is: {e - s} s")
