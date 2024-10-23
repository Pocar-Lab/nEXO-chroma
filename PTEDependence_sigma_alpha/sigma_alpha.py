from PocarChroma.geometry_manager import geometry_manager
from PocarChroma.run_manager import run_manager
from PocarChroma.material_manager import material_manager
from PocarChroma.surface_manager import surface_manager
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import argparse
import csv

def main(sigma_alpha):
    experiment_name = "8Silicon35_87"
    num_particles = 1_000_000
    seed = 1042
    visualize = True
    plots = []
    exclusion = []


    ptes = []
    ptes_err = []



    mm = material_manager(experiment_name=experiment_name)
    sm = surface_manager(material_manager = mm, experiment_name = experiment_name)
    sm.overwrite_property("silicon-Xe","sigma_alpha", sigma_alpha)

    gm = geometry_manager(
        experiment_name=experiment_name,
        visualize=visualize,
        exclude=exclusion,
        surf_manager=sm
    )

    rm = run_manager(
        geometry_manager=gm,
        experiment_name=experiment_name,
        random_seed=seed,
        num_particles=num_particles,
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
        writer.writerow([sigma_alpha, pte, ptes_err])



if __name__ == "__main__":
    for i in range(50):
        sigma_alpha = (i+1)/100
        main(sigma_alpha)
