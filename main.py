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
	print ("  	(1) '-n' <#>	number of photons to be simulated.") 
	print ("  	(2) '-s' <#>    choose the seed number")
	print ("  	(3) '-r' <#>    run ID number")
	print ("  Additional options can be chosen:")
	print ("  	(4) '-v'        to view the setup before simulation.")
	print ("=====================================================================")

def main():
	args = sys.argv[1:]
	try:
		opts, args = getopt.getopt(args,"vn:s:r:")
	except getopt.GetoptError:
		print ("  You have passed the wrong/or not enough arguments.")
		usage()
		sys.exit()
	n_opt = False
	s_opt = False
	r_opt = False
	v_opt = False
	for opt, arg in opts:
		if opt == '-n':
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
	
	if not n_opt:
		print("  Please provide number of particles, '-n' <#>")
		sys.exit()
	if not s_opt:
		print("  Please provide random seed, '-s' <#>")
		sys.exit()
	if not r_opt:
		print("  Please provide run ID, '-r' <#>")
		sys.exit()

	print('Number of particles:     ' + str(num_particles))
	print('Random seed:             ' + str(seed))
	print('Run ID:                  ' + str(run_id))

	gm = geometry_manager(run_id = run_id, visualize = v_opt)
	rm = run_manager(geometry_manager = gm, random_seed = seed, num_particles = num_particles, run_id = run_id)


if __name__ == '__main__':
	s = time.time()
	main()
	e = time.time()
	print(f'The simulation run time is: {e - s} s')
