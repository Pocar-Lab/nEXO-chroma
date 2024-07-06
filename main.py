#!/usr/bin/env python

import sys, getopt

from analysis.geometry_manager import geometry_manager
from analysis.run_manager import run_manager

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
    print ("  Additional options can be chosen:")
    print ("    (5) '-p' <Str1,Str2,...>    choose which plots to run")
    print ("  	(6) '-v'          to view the setup before simulation.")
    print ("  	(7) '-w'          to save simulation data from the run")
    print ("=====================================================================")

def main():
    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, "vn:s:r:e:p:w")
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
    write = False

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
            plots = [i.strip() for i in arg.split(',')]
        elif opt == '-w':
            write = True

    if not experiment_name:
        print("Please input an experiment name")
        usage()
        sys.exit()
    if not num_particles:
        print("Please input a number of particles")
        usage()
        sys.exit()
    if not seed:
        print("Please input a seed")
        usage()
        sys.exit()
    if not run_id:
        print(" Please input a run id")
        usage()
        sys.exit()

    print('Experiment Name:         ' + experiment_name)
    print('Number of particles:     ' + str(num_particles))
    print('Random seed:             ' + str(seed))
    print('Run ID:                  ' + str(run_id))
    print('Visualize:               ' + str(visualize))
    print('Plots:                   ' + ', '.join(plots))
    print('Saving Data:             ' + str(write) )


    gm = geometry_manager(experiment_name=experiment_name, run_id=run_id, visualize=visualize)
    rm = run_manager(geometry_manager=gm, experiment_name=experiment_name, random_seed=seed, num_particles=num_particles, run_id=run_id, plots=plots)



if __name__ == '__main__':
	s = time.time()
	main()
	e = time.time()
	print(f'The simulation run time is: {e - s} s')
