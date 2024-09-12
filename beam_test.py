#!/usr/bin/env python
from PocarChroma.geometry_manager import geometry_manager
from PocarChroma.run_manager import run_manager
from PocarChroma.run_manager import primary_generator

import time
from PocarChroma.document_manager import document_manager
import numpy as np

"""
Example showing how to run a simulation with a script file rather than from the command line.

Here perhaps include description of what the simulation is testing for archival reasons.

Run this file from within the Chroma container with `python ./ExampleAnalysis.py`
"""

def main():
    #experiment_name = "8refelctortall"
    experiment_name = "Sebastian_08.01.2023(liquefaction)_correctedSiPM" #define experiment
    LABEL = "beamtest" # label configuration or properties

    num_particles = 100000
    seed = 1042
    plots = [
            "plot_all_tracks",
             ]

    # e = [1, 2, 3, 4, 5, 6, 7, 8] #exclude outer
    e = [2, 3, 4, 5, 6, 7, 8, 9] #exclude outer

    # e = [9, 10, 11, 12, 13, 14, 15, 16] #exclude outer

    # e = [1, 3, 5, 7] #exclude outer

    e = [f"reflector{i}" for i in e]
    # e = None
    print(f"Experiment Name: {experiment_name}")
    print(f"Number of particles: {num_particles}")
    print(f"Random seed: {seed}")
    print(f"Plots: {plots}")
    print(f"Exclusions: {e}")

    gm = geometry_manager(
        experiment_name=experiment_name, exclude=e
    )

    gen = primary_generator(
        num_particles,
        center_pos=gm.get_solid_center(name="source"),
        source_type = "beam",
        beam_phi= np.pi,
        beam_theta = .04,
    )
    
    rm = run_manager(
        geometry_manager=gm,
        experiment_name=experiment_name,
        random_seed=seed,
        num_particles=num_particles,
        plots=plots,
        batches=False,
        pg=gen
    )
    print("Run manager complete")
    
    dm = document_manager(rm.ana_man, LABEL)
    dm.generate_typst_file()
    dm.compile_to_pdf()

if __name__ == "__main__":
    s = time.time()
    main()
    e = time.time()
    print(f"The simulation run time is: {e - s} s")
