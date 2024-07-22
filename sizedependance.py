#!/usr/bin/env python
from analysis.geometry_manager import geometry_manager
from analysis.run_manager import *
import time

import csv

"""
Example showing how to run a simulation with a script file rather than from the command line.

Here perhaps include description of what the simulation is testing for archival reasons.

Run this file from within the Chroma container with `python ./ExampleAnalysis.py`
"""


def main(r0):

    experiment_name = "SiliconeFlippedSourceLower"
    # experiment_name = "FlippedSourceTall"
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = False
    plots = []

    # exclusions = [[f"reflector{i}" for i in excl] for excl in exclusions]
    ptes = []
    ptes_err = []

    print(f"Experiment Name:       {experiment_name }")
    print(f"Number of particles:   {str(num_particles) }")
    print(f"Random seed:           {str(seed) }")
    print(f"Run ID:                {str(run_id) }")
    print(f"Visualize:             {str(visualize) }")
    print(f"Plots:                 {plots}")
    print(f"Source radius:         {r0}")
    # print(f"Excluded reflectors:   {exclusion}")

    gm = geometry_manager(
        experiment_name=experiment_name,
        run_id=run_id,
        visualize=visualize,
        exclude=[],
    )
    gen = primary_generator(
        num_particles,
        run_id=run_id,
        center_pos=gm.get_solid_center(name="source"),
        r=r0,
    )
    rm = run_manager(
        geometry_manager=gm,
        experiment_name=experiment_name,
        random_seed=seed,
        num_particles=num_particles,
        run_id=run_id,
        plots=plots,
        write=False,
        pg=gen,
    )
    pte = rm.ana_man.photon_transmission_efficiency
    ptes.append(pte)
    ptes_err.append(rm.ana_man.pte_st_dev)
    run_id += 1

    csv_file = "simulation_results.csv"
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([r0, pte, ptes_err[-1]])


import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ExampleAnalysis.py <r0>")
        sys.exit(1)
    r = float(sys.argv[1])
    s = time.time()
    main(r)
    e = time.time()
    print(f"The simulation run time is: {e - s} s")
