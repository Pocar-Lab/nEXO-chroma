#!/usr/bin/env python
from PocarChroma.geometry_manager import geometry_manager
from PocarChroma.run_manager import run_manager
import time
from PocarChroma.document_manager import document_manager

"""
Example showing how to run a simulation with a script file rather than from the command line.

Here perhaps include description of what the simulation is testing for archival reasons.

Run this file from within the Chroma container with `python ./ExampleAnalysis.py`
"""


def main():

    experiment_name = "SiliconeFlippedSourceLower"
    # experiment_name = "FlippedSourceTall"
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = False
    plots = [
        "plot_refl_angle",
        "plot_angle_hist",
        "plot_position_hist",
        # "plot_reflected_tracks",
        "plot_detected_tracks",
        "photon_shooting_angle",
        "photon_incident_angle_emission_angle_correlation",
    ]

    # exclusions = [ #flipped source tall
    #     [9, 10, 11, 12, 13, 14, 15, 16],
    #     [],
    #     [9, 10, 11, 12],
    #     [13, 14, 15, 16],
    # ]

    exclusions = [  # silicone flipped source lower
        # [2, 3, 4, 5, 6, 7, 8, 9],
        [],
        # [2, 3, 4, 5],
        # [
        #     6,
        #     7,
        #     8,
        #     9,
        # ],
    ]

    labels = [
        # "silicon-lower-noreflector",
        "silicon-lower-8reflector",
        # "silicon-lower-4reflector-outer",
        # "silicon-lower-4reflector-inner",
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
        dm = document_manager(rm.ana_man, labels[0])

        # Generate Typst file
        typst_filename = f"/workspace/results/{experiment_name}/report.typ"
        dm.generate_typst_file(typst_filename)

        # Compile Typst file to PDF
        pdf_filename = f"/workspace/results/{experiment_name}/report.pdf"
        dm.compile_to_pdf(typst_filename, pdf_filename)

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
