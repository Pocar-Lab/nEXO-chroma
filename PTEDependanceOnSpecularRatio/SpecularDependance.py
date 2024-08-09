from PocarChroma.geometry_manager import geometry_manager
from PocarChroma.run_manager import run_manager
from PocarChroma.material_manager import material_manager
from PocarChroma.surface_manager import surface_manager
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import argparse
import csv

def main(step_count, current_step):
    experiment_name = "SiliconeFlippedSourceUpper"
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = True
    plots = []

    exclusion = []
    label = "silicon-upper-8reflector"

    exclusions = [f"reflector{i}" for i in exclusion]  # map to name

    ptes = []
    ptes_err = []

    # Define the range
    start = 0
    end = 1
    step_size = (end - start) / (step_count - 1)
    specular = start + current_step * step_size

    print(f"Experiment Name:       {experiment_name }")
    print(f"Number of particles:   {str(num_particles) }")
    print(f"Random seed:           {str(seed) }")
    print(f"Run ID:                {str(run_id) }")
    print(f"Visualize:             {str(visualize) }")
    print(f"Plots:                 {plots}")
    print(f"Excluded reflectors:   {exclusion}")
    print(f"Specular Fraction:     {specular}")

    mm = material_manager(experiment_name=experiment_name,run_id=run_id)
    sm = surface_manager(material_manager = mm, experiment_name = experiment_name, run_id = run_id)
    sm.overwrite_property("silicon-Xe","reflect_specular", specular)
    sm.overwrite_property("silicon-Xe","reflect_diffuse", 1 - specular)

    gm = geometry_manager(
        experiment_name=experiment_name,
        run_id=run_id,
        visualize=visualize,
        exclude=exclusion,
        surf_manager=sm
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
    ptes_err = rm.ana_man.pte_st_dev

    print("Done!")

    print(ptes)
    print(ptes_err)

    # Append results to CSV
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([specular, pte, ptes_err])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment with a given specular fraction.")
    parser.add_argument("step_count", type=int, help="The total number of steps in the experiment.")
    parser.add_argument("current_step", type=int, help="The current step in the experiment.")
    args = parser.parse_args()
    main(args.step_count, args.current_step)
