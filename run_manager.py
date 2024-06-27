#!/usr/bin/env python

from chroma.sim import Simulation
from chroma.event import Photons

import numpy as np
import math
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from chroma import gpu

from analysis_manager import analysis_manager

class run_manager:
	def __init__(self, geometry_manager, experiment_name, random_seed, num_particles, run_id):
		self.num_steps = 100
		self.run_id = run_id
		self.seed = random_seed
		self.gm = geometry_manager
		self.center_pos = self.gm.get_solid_center(name = 'source')
		self.num_particles = num_particles
		self.sim = Simulation(self.gm.global_geometry, seed = random_seed, geant4_processes = 0)
		self.pg = primary_generator(self.num_particles, run_id = self.run_id, center_pos = self.center_pos)
		self.propagate_photon()
		self.ana_man = analysis_manager(self.gm, experiment_name, self.photons,  self.photon_tracks, self.run_id, self.seed, self.particle_histories)



# create primary generator which will create all the initial photons. we will tell primary generator
# what we want the source to be like
# get Photons from primary generator

	# def propagate_photon(self):
	# 	nthreads_per_block = 64
	# 	max_blocks = 1024
	# 	seed = 20000000

	# 	gpu_photons = gpu.GPUPhotons(self.pg.primary_photons)
	# 	gpu_geometry = gpu.GPUGeometry(self.gm.global_geometry)
	# 	self.photon_tracks = np.zeros((self.num_steps + 1, self.num_particles, 3))
	# 	self.photon_tracks[0, :, :] = self.pg.positions
	# 	rng_states = gpu.get_rng_states(nthreads_per_block * max_blocks, seed = seed)
	# 	for i in range(self.num_steps):
	# 		gpu_photons.propagate(gpu_geometry, rng_states, nthreads_per_block = nthreads_per_block, max_blocks = max_blocks, max_steps = 1)
	# 		self.photons = gpu_photons.get()
	# 		self.photon_tracks[i + 1, :, :] = self.photons.pos
	# 		# photon_track[i,:,0] = photons.pos[:,0] 
	# 		# photon_track[i,:,1] = photons.pos[:,1] 
	# 		# photon_track[i,:,2] = photons.pos[:,2]


	#where do we give value to photon_type, nr_steps, geometry, nthreads_per_block, max_blocks, rng_states?

	def propagate_photon(self):
		nthreads_per_block = 64   	
		max_blocks = 1024
		seed = 20000000

		gpu_photons = gpu.GPUPhotons(self.pg.primary_photons)
		gpu_geometry = gpu.GPUGeometry(self.gm.global_geometry)
		self.photon_tracks = np.zeros((self.num_steps + 1, self.num_particles, 3))
		self.photon_tracks[0, :, :] = self.pg.positions
		rng_states = gpu.get_rng_states(nthreads_per_block * max_blocks, seed = seed)
		self.interactions = {'RAYLEIGH_SCATTER': 4,
							 'REFLECT_DIFFUSE': 5, 
							 'REFLECT_SPECULAR': 6,
							 'SURFACE_REEMIT': 7,
							 'SURFACE_TRANSMIT': 8,
							 'BULK_REEMIT': 9,
							 'CHERENKOV': 10,
							 'SCINTILLATION': 11}
		self.particle_histories = {curr_int: np.zeros(self.num_particles, dtype = int) for curr_int in self.interactions.keys()}

		for i in range(self.num_steps):
			# propagate photons
			gpu_photons.propagate(gpu_geometry, rng_states, nthreads_per_block = nthreads_per_block, max_blocks = max_blocks, max_steps = 1)
			# from GPUPhoton get event.Photon object
			self.photons = gpu_photons.get()
			self.photon_tracks[i + 1, :, :] = self.photons.pos
			# collect information from flags about interaction history
			self.update_tallies(self.photons)
			# reset interaction history bits that are nonterminal
			new_flags = self.reset_nonterminal_flags(self.photons.flags)
			gpu_photons.flags[:self.num_particles].set(new_flags.astype(np.uint32))
			

	def reset_nonterminal_flags(self, flag_list):
		# mask = 1111111111111111111000000001111
		mask = 2147479567
		return flag_list & mask

	def update_tallies(self, photons):
		for key, value in self.interactions.items():
			curr_tally = (photons.flags & (0x1 << value)).astype(bool).astype(int)
			self.particle_histories[key] += curr_tally
			# print(key, self.particle_histories[key])



class primary_generator: #photon generator
	# C++: methods/functions
	# def __init__(self, num_particles, center_pos = [0, 0, 0], delta_placement = 0.0):
	def __init__(self, num_particles, run_id, center_pos = [0, 0, 0]):
		self.num_particles = num_particles
		self.center_pos = center_pos


		# positions


		self.positions = self.get_point_source_pos(x0 = self.center_pos[0],
											y0 = self.center_pos[1],
											z0 = self.center_pos[2])

		# self.positions = self.get_xz_disk_source_pos(x0 = self.center_pos[0],
		# 									y0 = self.center_pos[1],
		# 									z0 = self.center_pos[2],
		# 									r = 2.5)

		# beam_dir = self.get_beam_dir(px = 0, py = 1, pz = 0)# The beam travel in y direction
		
		# from 0705, try to define the angle of the beam_direction rather than just in y

	    # define the beam angle from run_id

		# angle = np.pi * (61.5+0.1*(run_id)) / 180
		# angle = np.pi * 60.5/ 180
		# phi_angle = np.pi*45 / 180
		# beam_dir = self.get_beam_dir(angle)
		# print("the beam angle in degree is", angle*180 / np.pi)

		# x_cone_dir = self.get_x_cone_dir(angle = np.pi, positive = False)

		# y_cone_dir = self.get_y_cone_dir(angle = np.pi / 32.0, positive = True)

		# curr_dir = y_cone_dir
		isotropic_dir = self.get_isotropic_dir()
		# self.directions = y_cone_dir
		# self.directions = beam_dir
		self.directions = isotropic_dir

		# fig = plt.figure()
		# ax = plt.axes(projection = '3d')

		# ax.scatter(curr_dir[:, 0], curr_dir[:, 1], curr_dir[:, 2])
		# ax.set_xlim(-1, 1)
		# ax.set_ylim(-1, 1)
		# ax.set_zlim(-1, 1)
		# plt.xlabel('x')
		# plt.ylabel('y')

		# plt.show(block = True)

		# polarization
		self.polarization = np.cross(self.directions, self.get_isotropic_dir())
		# wavelengths
		self.wavelengths = np.ones(self.num_particles) * 175.0
		# create photons
		self.primary_photons = Photons(self.positions,
									self.directions,
									self.polarization,
									self.wavelengths)

	def get_point_source_pos(self, x0, y0, z0):
		return np.tile([x0, y0, z0], (self.num_particles, 1))

	# input center of disk and radius
	def get_xy_disk_source_pos(self, x0, y0, z0, r):
		# pos_array = np.empty(self.num_particles, 3)
		curr_sqrtr = np.sqrt(np.random.uniform(0, r, self.num_particles))
		curr_theta = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
		curr_x = curr_sqrtr * np.cos(curr_theta) + x0
		curr_y = curr_sqrtr * np.sin(curr_theta) + y0
		curr_z = np.ones(self.num_particles) * z0
		return np.vstack((curr_x, curr_y, curr_z)).T
	def get_xz_disk_source_pos(self, x0, y0, z0, r):
		# pos_array = np.empty(self.num_particles, 3)
		curr_sqrtr = np.sqrt(np.random.uniform(0, r, self.num_particles))
		curr_theta = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
		curr_x = curr_sqrtr * np.cos(curr_theta) + x0
		curr_y = np.ones(self.num_particles) * y0
		curr_z = curr_sqrtr * np.sin(curr_theta) + z0
		return np.vstack((curr_x, curr_y, curr_z)).T

	# def get_yz_disk_source_pos(self, x0, y0, z0, r):

	# def get_xy_cylinder_source_pos(self, x0, y0, z0, r, h):


	def get_isotropic_dir(self):
		phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
		# phi = np.random.uniform(0, np.pi, self.num_particles)
		# mu = cos(theta)
		cos_theta = np.random.uniform(-1.0, 1.0, self.num_particles)
		sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

		curr_px = np.cos(phi) * sin_theta
		curr_py = np.sin(phi) * sin_theta
		curr_pz = cos_theta
		# print(np.shape(curr_pz))
		# theta = math.acos(curr_pz)
		# emit_angle = (0.5 * np.pi) - theta
		# self.get_emission_angle(emit_angle)
		return np.vstack((curr_px, curr_py, curr_pz)).T


	# #07/11 get emission angle
	# def get_emission_angle(self, emission_angle):
	# 	self.emission_angle = emission_angle
	# 	print("emission angle is", self.emission_angle)
	# 	return self.emission_angle
		
    # 07/05, get the beam move in xy plane in +/- x direction;
	def get_beam_dir(self, angle):
		curr_py = np.cos(angle)
		curr_px = np.sin(angle)
		# # curr_pz = 0 shifting the beam in xy plane
		curr_pz = 0 
		# curr_pz = np.sin(angle)*np.cos(phi_angle)
		# curr_px = np.sin(angle)*np.sin(phi_angle)
		return np.tile([curr_px, curr_py, curr_pz], (self.num_particles, 1))

	def get_x_cone_dir(self, angle, positive = True):
		phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
		cos_theta = np.random.uniform(np.cos(angle/2), 1, self.num_particles)
		sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

		curr_px = cos_theta
		curr_py = np.cos(phi) * sin_theta
		curr_pz = np.sin(phi) * sin_theta
		if not positive:
			curr_px *= -1
		return np.vstack((curr_px, curr_py, curr_pz)).T

	def get_y_cone_dir(self, angle, positive = True):
		phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
		cos_theta = np.random.uniform(np.cos(angle/2), 1, self.num_particles)
		sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

		curr_px = np.cos(phi) * sin_theta
		curr_py = cos_theta
		curr_pz = np.sin(phi) * sin_theta
		if not positive:
			curr_py *= -1

		# print(np.vstack((curr_px, curr_py, curr_pz)).T)
		return np.vstack((curr_px, curr_py, curr_pz)).T
		


	# for placement:
	# -feed in specific geometry component name and use the center as the location for photons
	# -feed in specific geometry component name and use entire surface (for a disk case)
	# -feed in specific (x, y, z) point
	# -feed in disk with center (x, y, z) and radius r
	# -feed in cylinder with center (x, y, z) with radius r and height h

	# for direction
	# -isotropic (equal probability in all directions) half space?
	# -cone defined by angle theta and mu direction
	# -beam defined by mu direction
	# -only simulate half space/reduced space?

	# create Photons(position_array,direction_array,polarization_array,wavelengths_array)

	



