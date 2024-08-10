from analysis.geometry_manager import geometry_manager
from analysis.run_manager import run_manager
from analysis.run_manager import primary_generator
from analysis.surface_manager import surface_manager
from analysis.material_manager import material_manager
import time
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.umath as umath
def main():

    experiment_name = "8Silicon35_87"
    num_particles = 1_000_000
    seed = 104
    run_id = 1
    visualize = False
    plots = []
    write = False

    absorption_lengths = []
    ptes = []
    model_ptes = []
    ptes_err = []
    D = 39.98
    # D = 35.87
    distance_arr = []

    def calc_distance(pte_lambda,pte_no_absorption,absorption_length):
        return -absorption_length * umath.log(pte_lambda/pte_no_absorption)
    

    mm = material_manager(experiment_name=experiment_name,run_id=run_id)
    mm.add_attributes(mm.get_material("liquid xenon"), absorption_length = 1_000_000_000_000)
    mm.add_attributes(mm.get_material("silica"), absorption_length = 1_000_000_000_000)

    sm = surface_manager(material_manager = mm, experiment_name = experiment_name, run_id = run_id)
    gm = geometry_manager(experiment_name=experiment_name, run_id=run_id, visualize=visualize, surf_manager = sm)

    gen = primary_generator(
        num_particles,
        run_id=run_id,
        center_pos=gm.get_solid_center(name="source"),
        source_type = "beam"
    )
    rm = run_manager(geometry_manager=gm, experiment_name=experiment_name, random_seed=seed, num_particles=num_particles, run_id=run_id, plots=plots, write = write, pg = gen)
    pte_no_absorption = unc.ufloat(rm.ana_man.photon_transmission_efficiency, rm.ana_man.pte_st_dev)
    for i in range(30):
        absorption_length = 5 * (i+1)
        mm = material_manager(experiment_name=experiment_name,run_id=run_id)
        mm.add_attributes(mm.get_material("liquid xenon"), absorption_length = absorption_length)
        mm.add_attributes(mm.get_material("silica"), absorption_length = absorption_length)
        mm.add_attributes(mm.get_material("liquid xenon"), refractive_index = 1)

        mm.add_attributes(mm.get_material("silica"), refractive_index = 1)

        sm = surface_manager(material_manager = mm, experiment_name = experiment_name, run_id = run_id)
        gm = geometry_manager(experiment_name=experiment_name, run_id=run_id, visualize=visualize, surf_manager = sm)

        gen = primary_generator(
        num_particles,
        run_id=run_id,
        center_pos=gm.get_solid_center(name="source"),
        source_type = "beam"
        )
        rm = run_manager(geometry_manager=gm, experiment_name=experiment_name, random_seed=seed, num_particles=num_particles, run_id=run_id, plots=plots, write = write, pg = gen)

        pte_lambda = unc.ufloat(rm.ana_man.photon_transmission_efficiency, rm.ana_man.pte_st_dev)


        # distance = -absorption_length * np.log((pte)/pte_no_absorption)
        distance = calc_distance(pte_lambda, pte_no_absorption, absorption_length)
        distance_arr.append(distance)
    
    # ufloats = [unc.ufloat(ptes[i],ptes_err[i]) for i in range(len(ptes))]
    # print(ufloats)
    # plt.plot(absorption_lengths, ptes)
    # plt.scatter(absorption_lengths, model_ptes,)

    # plt.show()
    # print(distance_arr)
    distance_arr = np.array(distance_arr)
    print(distance_arr.mean())


def calculate_distance(pte,pte_inf,absorption_length):
     return -absorption_length * unc.umath.log(pte/pte_inf)
if __name__ == '__main__':
	s = time.time()
	e = main()
	# e = time.time()
	# print(f'The simulation run time is: {e - s} s')