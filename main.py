#!/usr/bin/env python

import sys, getopt

from geometry_manager import geometry_manager
from run_manager import run_manager

import time

# -n number of particles
# -s random seed
# -v visualize
def usage():
	print ("=====================================================================")
	print ("  The minimum paramaters the simulation needs are:")
	print ("    (1) '-e' <Str>              name of experiment to be simulated.")
	print ("  	(2) '-n' <#>	            number of photons to be simulated.") 
	print ("  	(3) '-s' <#>                choose the seed number")
	print ("  	(4) '-r' <#>                run ID number")
	print ("    (5) '-p' <Str1,Str2,...>    choose which plots to run")
	print ("  Additional options can be chosen:")
	print ("  	(6) '-v'          to view the setup before simulation.")
	print ("=====================================================================")

def main():
    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, "vn:s:r:e:p:")
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        usage()
        sys.exit()

    experiment_name = None
    num_particles = None
    seed = None
    run_id = None
    visualize = False
    plots = []

    for opt, arg in opts:
        if opt == '-e':
            experiment_name = str(arg)
        elif opt == '-n':
            num_particles = int(arg)
        elif opt == '-s':
            seed = int(arg)
        elif opt == '-r':
            run_id = int(arg)
        elif opt == '-v':
            visualize = True
        elif opt == '-p':
            plots = arg.split(',')

    if not experiment_name or not num_particles or not seed or not run_id or len(plots) == 0:
        print("  You have passed the wrong/or not enough arguments.")
        usage()
        sys.exit()

    print('Experiment Name:         ' + experiment_name)
    print('Number of particles:     ' + str(num_particles))
    print('Random seed:             ' + str(seed))
    print('Run ID:                  ' + str(run_id))
    print('Visualize:               ' + str(visualize))
    print('Plots:                   ' + ', '.join(plots))

    gm = geometry_manager(experiment_name=experiment_name, run_id=run_id, visualize=visualize)
    rm = run_manager(geometry_manager=gm, experiment_name=experiment_name, random_seed=seed, num_particles=num_particles, run_id=run_id, plots=plots)



if __name__ == '__main__':
	s = time.time()
	main()
	e = time.time()
	print(f'The simulation run time is: {e - s} s')
