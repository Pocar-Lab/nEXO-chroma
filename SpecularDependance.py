from analysis.geometry_manager import geometry_manager
from analysis.run_manager import run_manager
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

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
speculars = np.linspace(0, 1, 2)
for specular in speculars:
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

    # set ratios of specular vs diffuse
    gm.surf_manager.surfaces["silicon-Xe"].set("reflect_specular", specular)

    #gm.surf_manager.surfaces["silicon"].set("reflect_specular", specular)
    #gm.surf_manager.surfaces["silicon"].set("reflect_diffuse", 1 - specular)

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
print("Done!")

print(ptes)
print(ptes_err)

plt.figure(figsize=(15, 10))
plt.errorbar(speculars, ptes, ptes_err, capsize=5)
plt.xlabel("Specular Fraction")
plt.ylabel("PTE")
plt.title("PTE as Function of Specular Ratio")
plt.grid()
plt.savefig("./output.png")