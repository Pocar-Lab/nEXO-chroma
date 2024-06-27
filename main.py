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
	print ("    (1) '-e' <Str>  name of experiment to be simulated.")
	print ("  	(2) '-n' <#>	  number of photons to be simulated.") 
	print ("  	(3) '-s' <#>      choose the seed number")
	print ("  	(4) '-r' <#>      run ID number")
	print ("  Additional options can be chosen:")
	print ("  	(5) '-v'          to view the setup before simulation.")
	print ("=====================================================================")

def main():
	args = sys.argv[1:]
	try:
		opts, args = getopt.getopt(args,"vn:s:r:e:")
	except getopt.GetoptError:
		print ("  You have passed the wrong/or not enough arguments.")
		usage()
		sys.exit()
	e_opt = False
	n_opt = False
	s_opt = False
	r_opt = False
	v_opt = False
	for opt, arg in opts:
		if opt == '-e':
			experiment_name = str(arg)
			e_opt = True
		elif opt == '-n':
			num_particles = int(arg)
			n_opt = True
		elif opt == '-s':
			seed = int(arg)
			s_opt = True
		elif opt == '-r':
			run_id = int(arg)
			r_opt = True
		elif opt == '-v':
			v_opt = True
	if not e_opt:
		print("  Please provide the name of the simulation to be run, '-e' <String>")
	if not n_opt:
		print("  Please provide number of particles, '-n' <#>")
		sys.exit()
	if not s_opt:
		print("  Please provide random seed, '-s' <#>")
		sys.exit()
	if not r_opt:
		print("  Please provide run ID, '-r' <#>")
		sys.exit()

	print('Experiment Name:         ' + experiment_name)
	print('Number of particles:     ' + str(num_particles))
	print('Random seed:             ' + str(seed))
	print('Run ID:                  ' + str(run_id))

	gm = geometry_manager(experiment_name = experiment_name, run_id = run_id, visualize = v_opt)
	rm = run_manager(geometry_manager = gm, experiment_name = experiment_name, random_seed = seed, num_particles = num_particles, run_id = run_id)


if __name__ == '__main__':
	s = time.time()
	main()
	e = time.time()
	print(f'The simulation run time is: {e - s} s')
