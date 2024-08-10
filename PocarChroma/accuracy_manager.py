import numpy as np
import matplotlib.pyplot as plt
from geometry_manager import geometry_manager
from run_manager import run_manager
from material_manager import material_manager
from surface_manager import surface_manager


def getPTE(runlabel):
    experiment_name = None
    num_particles = 1_000_000
    seed = 1042
    run_id = 1
    visualize = True
    plots = []

    print(f"Experiment Name:       {experiment_name }")
    print(f"Number of particles:   {str(num_particles) }")
    print(f"Random seed:           {str(seed) }")
    print(f"Run ID:                {str(run_id) }")
    print(f"Visualize:             {str(visualize) }")
    print(f"Plots:                 {plots}")

    gm = geometry_manager(
        experiment_name=experiment_name, run_id=run_id, visualize=visualize
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
    return 0


def getGamma():
    """
    Calculate the Gamma matrix, a matrix of PTE run ratios based on simulation.
    """
    runs = []
    Gamma = []
    ptes = [getPTE(run) for run in runs]
    Gamma = np.zeros((len(ptes), len(ptes)))
    for i, j in zip(range(len(ptes)), range(len(ptes))):
        Gamma[i, j] = ptes[i] / ptes[j]
    return Gamma


def getLambda():
    """
    Get Lambda, a matrix of the alpha amplitude ratios based on experimental data.
    """

    amplitudes = []  # TODO load from file
    Lambda = np.zeros((len(amplitudes), len(amplitudes)))
    for i, j in zip(range(len(amplitudes)), range(len(amplitudes))):
        Lambda[i, j] = amplitudes[i] / amplitudes[j]
    return Lambda


def getScore(Gamma, Lambda):
    return np.linalg.norm(Gamma - Lambda, ord="fro")
