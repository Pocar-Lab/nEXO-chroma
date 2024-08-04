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
    #experiment_name = "8refelctortall"
    experiment_name = "Sebastian_08.01.2023(liquefaction)_correctedSiPM"
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = False
    plots = [
            # "plot_all_tracks" ,
            # "plot_detected_tracks" ,
            # "plot_undetected_tracks" ,
            # "plot_reflected_tracks" ,
            # "plot_filtered_scattered_tracks" ,
            # "plot_detected_reflected_tracks" ,
            # "plot_specular_reflected_tracks" ,
            # "plot_diffuse_reflected_tracks" ,
            # "plot_refl_multiplicity" ,
            # "photon_shooting_angle" ,
            # "photon_incident_angle_emission_angle_correlation" ,
            # "plot_angle_hist" ,
            # "plot_refl_angle" ,
            # "plot_position_hist" ,
             ]
    # exclusions = [ #8refelctor
    #     [1, 2, 3, 4, 5, 6, 7, 8],
    #     [],
    #     [1, 3, 5, 7],
    #     [2, 4, 6, 8],
    #     [1, 3, 4, 5, 7, 8],
    #     [1, 2, 3, 5, 6, 7],
    #     [ 2, 3, 4, 6, 7, 8],
    #     [1, 2, 4, 5, 6, 8],
    # ]

    exclusions = [ #8/01
        [ 2, 3, 4, 5, 6, 7, 8, 9],
        [],
        [8, 4, 6, 7], #outer
        [2, 5, 3, 9], #inner
        [ 3, 4, 6, 7, 8, 9], #2-5
        [ 2, 4, 5, 6, 7, 8], #3-9
         [ 2, 3, 5, 7, 8, 9], #4-6
          [ 2, 3, 4, 5, 6, 9], #8-7
    ]
    excl = [[f"reflector{i}" for i in e] for e in exclusions]

    LABELS = [
        "silicon-short-no-reflector    ",
        "silicon-short-8reflector      ",
        "silicon-short-4reflector-outer",
        "silicon-short-4reflector-inner",
        "silicon-short-2reflector-2-5  ",
        "silicon-short-2reflector-3-9  ",
        "silicon-short-2reflector-4-6  ",
        "silicon-short-2reflector-8-7  ",

    ]
    PTEs = []
    PTEerrs = []
    for e, label in zip(excl, LABELS):
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
        )
        PTE = rm.ana_man.photon_transmission_efficiency
        PTEerr = rm.ana_man.pte_st_dev_exp
        PTEs.append(PTE)
        PTEerrs.append(PTEerr)
    # dm = document_manager(rm.ana_man, LABEL)
    # dm.generate_typst_file()
    # dm.compile_to_pdf()
    print("Conclusion")
    print("------------")
    for i in range(len(LABELS)):
        label = LABELS[i]
        pte = PTEs[i]
        PTEerr = PTEerrs[i]
        print(f"{label}: {pte} +- {PTEerr}")
    print(LABELS)
    print(PTEs)
    print(PTEerrs)

if __name__ == "__main__":
    s = time.time()
    main()
    e = time.time()
    print(f"The simulation run time is: {e - s} s")
