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

def main():
    #experiment_name = "8refelctortall"
    experiment_name = "8Silicon35_87" #define experiment
    LABEL = "NO REFLECTOR" # label configuration or properties

    num_particles = 100_000_000
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

    e = [1, 2, 3, 4, 5, 6, 7, 8] #exclude outer
    print(f"Experiment Name: {experiment_name}")
    print(f"Number of particles: {num_particles}")
    print(f"Random seed: {seed}")
    print(f"Run ID: {run_id}")
    print(f"Visualize: {visualize}")
    print(f"Plots: {plots}")
    print(f"Exclusions: {e}")

    gm = geometry_manager(
        experiment_name=experiment_name, run_id=run_id, visualize=visualize, exclude=e
    )
    rm = run_manager(
        geometry_manager=gm,
        experiment_name=experiment_name,
        random_seed=seed,
        num_particles=num_particles,
        run_id=run_id,
        plots=plots,
        batches=True
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
