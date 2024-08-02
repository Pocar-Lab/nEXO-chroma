#!/usr/bin/env python
from analysis.geometry_manager import geometry_manager
from analysis.run_manager import run_manager
import time
from analysis.document_manager import document_manager

"""
Example showing how to run a simulation with a script file rather than from the command line.

Here perhaps include description of what the simulation is testing for archival reasons.

Run this file from within the Chroma container with `python ./ExampleAnalysis.py`
"""


import multiprocessing
from functools import partial

def run_simulation(i, experiment_name, num_particles, seed, run_id, visualize, plots):
    print(f"Starting simulation {i}")
    
    print(f"Finished simulation {i}")

def main():
    experiment_name = "8refelctortall"
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = False
    plots = [
            "plot_all_tracks" ,
            "plot_detected_tracks" ,
            "plot_undetected_tracks" ,
            "plot_reflected_tracks" ,
            "plot_filtered_scattered_tracks" ,
            "plot_detected_reflected_tracks" ,
            "plot_specular_reflected_tracks" ,
            "plot_diffuse_reflected_tracks" ,
            "plot_refl_multiplicity" ,
            "photon_shooting_angle" ,
            "photon_incident_angle_emission_angle_correlation" ,
            "plot_angle_hist" ,
            "plot_refl_angle" ,
            "plot_position_hist" ,
             ]
    excl = [1, 3, 5, 7, 2, 4, 6, 8]
    LABEL = "silicon-noreflector-tall"
    print(f"Experiment Name: {experiment_name}")
    print(f"Number of particles: {num_particles}")
    print(f"Random seed: {seed}")
    print(f"Run ID: {run_id}")
    print(f"Visualize: {visualize}")
    print(f"Plots: {plots}")

    gm = geometry_manager(
        experiment_name=experiment_name, run_id=run_id, visualize=visualize, exclude=[f"reflector{i}" for i in excl]
    )
    rm = run_manager(
        geometry_manager=gm,
        experiment_name=experiment_name,
        random_seed=seed,
        num_particles=num_particles,
        run_id=run_id,
        plots=plots,
    )
    dm = document_manager(rm.ana_man, LABEL)
    dm.generate_typst_file()
    dm.compile_to_pdf()

if __name__ == "__main__":
    s = time.time()
    main()
    e = time.time()
    print(f"The simulation run time is: {e - s} s")
