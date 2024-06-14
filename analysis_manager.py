#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits import mplot3d
from array import array


class analysis_manager:
	def __init__(self, geometry_manager, photons, photon_tracks = 1000, run_id = 0, seed = 0, histories = None):
		self.gm = geometry_manager
		self.photons = photons
		self.photon_tracks = photon_tracks
		self.num_particles = len(self.photons)
		self.run_id = run_id
		self.seed = seed
		self.particle_histories = histories
		self.get_tallies()
		num_tracks = 100000

		# self.plot_tracks(num_tracks = num_tracks, detected_only = True, reflected_only = True)
		# self.plot_tracks_reed_rewrite(num_tracks = None, detected_only = True, reflected_only = False)
		# self.plot_undetected_tracks(num_tracks = 10000)
		# self.plot_filtered_tracks(filter_by = 5, num_tracks = num_tracks)
		# self.plot_filtered_scattered_tracks(num_tracks = num_tracks)
		# self.plot_refl_multiplicity(density = True)
		# self.photon_shooting_angle(num_tracks = None, detected_only = True, reflected_only = False)		
		# self.photon_incident_angle_emission_angle_correlation(num_tracks = None, detected_only = True, reflected_specular_only = False, reflected_diffuse_only = False)
		# histogramfilename = '/workspace/results/data/Sebastian_teflon_05.23.2023/histogram_teflon/spec_tef_histogram_' + str(self.run_id) + '.csv'
		histogramfilename = '/workspace/results/data/Sebastian_FS_06.08.2023_correctedSiPM/histogram_teflon/diff_spec_histogram_'+ str(self.run_id) + '.csv'
		# histogramfilename ='/workspace/results/data/Sebastian_08.01.2023(liquefaction)_correctedSiPM/histogramPd_histogram_0.csv'
		self.plot_angle_hist(histogramfilename)
		self.plot_refl_angle(low_angle = 12, do_log = False)
		self.plot_position_hist()
		#self.plot_efficiency_R_I()                        #uncommenting this leads to error
		# filename = '/workspace/results/data/copperplates_06.23.2022/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/copperplates_06.23.2022/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/beam_direction_06.30.2022/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/silica_window_07.18.2022/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/source_copperholder_08.16.2022/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/copper_gasket_08.29.2022/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/Al_filler_02.07.2023/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/sourcepart_05.11.2023/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/Sebastian_teflon_05.23.2023/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/Sebastian_woteflon_05.23.2023/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		# filename = '/workspace/results/data/Sebastian_flippedsource_06.06.2023/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'		
		# filename = '/workspace/results/data/Sebastian_FS_06.08.2023_correctedSiPM/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		filename = '/workspace/results/data/Sebastian_08.01.2023(liquefaction)_correctedSiPM/datapoints/hd3_data_test_' + str(self.run_id) + '.csv'
		self.save_detected(filename)
		# self.temperory_histogram(filename)


	def incident_angle(self, last_pos):
		angles = np.arccos(np.fabs(last_pos[:,1])/np.sqrt((last_pos[:,0]**2 + last_pos[:,1]**2 + last_pos[:,2]**2)))*(180./np.pi)
		return angles

	def get_tallies(self):
		self.tallies = {}
		self.tallies['NO_HIT']           = (self.photons.flags & (0x1 << 0)).astype(bool)
		self.tallies['BULK_ABSORB']      = (self.photons.flags & (0x1 << 1)).astype(bool)
		self.tallies['SURFACE_DETECT']   = (self.photons.flags & (0x1 << 2)).astype(bool)
		self.tallies['SURFACE_ABSORB']   = (self.photons.flags & (0x1 << 3)).astype(bool)
		##place holder below, the output value doesn't mean anything.
		# self.tallies['RAYLEIGH_SCATTER'] = (self.photons.flags & (0x1 << 4)).astype(bool)
		self.tallies['REFLECT_DIFFUSE']  = (self.photons.flags & (0x1 << 5)).astype(bool)
		# self.tallies['REFLECT_SPECULAR'] = (self.photons.flags & (0x1 << 6)).astype(bool)
		# self.tallies['SURFACE_REEMIT']   = (self.photons.flags & (0x1 << 7)).astype(bool)
		# self.tallies['SURFACE_TRANSMIT'] = (self.photons.flags & (0x1 << 8)).astype(bool)
		# self.tallies['BULK_REEMIT']      = (self.photons.flags & (0x1 << 9)).astype(bool)
		# self.tallies['CHERENKOV']        = (self.photons.flags & (0x1 << 10)).astype(bool)
		# self.tallies['SCINTILLATION']    = (self.photons.flags & (0x1 << 11)).astype(bool)
		# self.tallies['NAN_ABORT']        = (self.photons.flags & (0x1 << 31)).astype(bool)


		print()
		print()
		print('--SUMMARY---------------------------')
		print('NUM_PARTICLES', self.num_particles)
		for key, value in self.tallies.items():
			print(key, np.sum(value))
		for key, value in self.particle_histories.items():
			print(key, np.sum(value.astype(bool)), 'at least once')
			print(key, np.sum(value), 'total number')

		print()
		self.efficiency = np.sum(self.tallies['SURFACE_DETECT']) / self.num_particles
		print('EFFICIENCY', self.efficiency)
		print('------------------------------------')

		self.detected_positions = self.photons.pos[self.tallies['SURFACE_DETECT']]
		# print('the detected positions in x, y, z is', self.detected_positions)
		self.detected_angles = self.incident_angle(self.photons.dir[self.tallies['SURFACE_DETECT']])


	def save_detected(self, filename):
		# lxe_refractive_index = self.gm.mat_manager.get_material('liquid xenon').refractive_index[0, 1]
		lxe_refractive_index = self.gm.mat_manager.material_props['liquid xenon']['refractive_index']
		# print(lxe_refractive_index,'in saved detected')

		# silica_refractive_index = self.gm.mat_manager.material_props['silica']['refractive_index']
		# Cu_eta_index = self.gm.mat_manager.material_props['copper']['eta']
		# Cu_k_index = self.gm.mat_manager.material_props['copper']['k']
		Si_eta_index = self.gm.mat_manager.material_props['silicon']['eta']
		Si_k_index = self.gm.mat_manager.material_props['silicon']['k']
		# al_eta = self.gm.mat_manager.material_props['aluminum']['eta']
		# al_k = self.gm.mat_manager.material_props['aluminum']['k']
		# Cu_spec_coefficient = self.gm.surf_manager.surface_props['Cu-Xe']




	    # store PTE for both 0 and non zero
		self.emit_angle = self.run_id - 1
		if self.efficiency != 0:
			# print("self.efficiency is", self.efficiency)
			x_detected_position = self.detected_positions[:, 0]
			# print(len(x_detected_position))
			# print(type(x_detected_position))
			y_detected_position = self.detected_positions[:, 1]
			z_detected_position = self.detected_positions[:, 2]
			detected_angle = self.detected_angles
		else:
			x_detected_position = "none"
			y_detected_position = "none"
			z_detected_position = "none"
			detected_angle = "none"		
		save_data = {
				'x (mm)': x_detected_position,
				'y (mm)': y_detected_position,
				'z (mm)': z_detected_position,
				'emission angle': self.emit_angle,
				'angle': detected_angle,
				'PTE': np.array([self.efficiency for _ in range(len(detected_angle))]),
				'LXe Index of Refraction': np.array([lxe_refractive_index for _ in range(len(detected_angle))]),
				'Seed number': np.array([self.seed for _ in range(len(detected_angle))]),
				'Si eta': np.array([Si_eta_index for _ in range(len(self.detected_angles))]),
				'Si k': np.array([Si_k_index for _ in range(len(self.detected_angles))])}
				# 'Cu spec coeff': np.array([Cu_spec_coefficient for _ in range(len(self.detected_angles))])}

				# 'Fused Silica Index of Refraction': np.array([silica_refractive_index for _ in range(len(self.detected_angles))]),
				# 'Cu eta': np.array([cu_eta for _ in range(len(self.detected_angles))]),
				# 'Cu k': np.array([cu_k for _ in range(len(self.detected_angles))]),
				# 'Al eta': np.array([al_eta for _ in range(len(self.detected_angles))]),
				# 'Al k': np.array([al_k for _ in range(len(self.detected_angles))])
		# print(len(save_data))
		df = pd.DataFrame(save_data)
		df.to_csv(filename)


	def temperory_histogram(self,filename):
		exclude_incidentangle_file = '/workspace/results/data/Sebastian_teflon_05.23.2023/datapoints/exclude_incident_angle_' + str(self.run_id) + '.csv'
		colume_incidentangle = ['angle']
		colume_y = ['y (mm)']
		incident_angle = pd.read_csv(filename, usecols=colume_incidentangle).to_numpy()
		# print('incident angle saved?',len(incident_angle))
		y_position = pd.read_csv(filename, usecols=colume_y).to_numpy()
		print('min detected y',y_position.min())
		print('max detected y',y_position.max())
		# print('number of y position', len(y_position))
		real_inc_angle=[]
		exclude_inc_angle=[]
		exclude_y=[]
		for i in range(len(y_position)):
			if y_position[i] <= 56.972675:
				# print(incident_angle[i])
				real_inc_angle.append(float(incident_angle[i]))
			else:
				exclude_inc_angle.append(float(incident_angle[i]))
				exclude_y.append(float(y_position[i]))

		# print(type(np.array(real_inc_angle)))
		print(len(real_inc_angle))
		print(len(exclude_inc_angle))
		save_exclude_inc_angle = {
							'y (mm)': np.array(exclude_y),
							'incident angle': np.array(exclude_inc_angle)}
		print(len(save_exclude_inc_angle))

		df = pd.DataFrame(save_exclude_inc_angle)
		df.to_csv(exclude_incidentangle_file)	

		fig = plt.figure()
		plt.hist(real_inc_angle, bins = [theta for theta in range(91)])

		hist, bin_edges = np.histogram(real_inc_angle, bins = [theta for theta in range(91)])
		print(hist)

		plt.title('Incident angle histogram(corrected)')
		plt.xlabel('Incident angle[degree]')
		plt.ylabel('Counts')
		plt.show()

	def plot_tracks_reed_rewrite(self, num_tracks = None, detected_only = True, specular_only = False, diffuse_only = False):
		if num_tracks == None:
			num_tracks = self.num_particles

		fig = plt.figure()
		ax = plt.axes(projection = '3d')

		mask = np.ones(self.num_particles, dtype=bool)

		if detected_only:
			mask &= self.tallies['SURFACE_DETECT']

		if specular_only:
			mask &= self.particle_histories['REFLECT_SPECULAR'].astype(bool) 
				
		if diffuse_only:
			mask &= self.particle_histories['REFLECT_DIFFUSE'].astype(bool)

		filtered_tracks = self.photon_tracks[:, mask, :]
		print(np.shape(filtered_tracks))
		save_tracks = []
		_, num_good_tracks, _ = np.shape(filtered_tracks)
		if num_good_tracks < num_tracks:
			num_tracks = num_good_tracks
		for i in range(num_tracks):
			track = filtered_tracks[:, i, :]
			unique_ind = track[:-1, :] != track[1:, :]
			unique_ind = list(np.sum(unique_ind, axis = 1) > 0)
			unique_ind = [True] + unique_ind
			track = track[unique_ind, :]
			save_tracks.append(track)
			ax.plot(track[:, 0], track[:,1], track[:,2], color = 'tab:blue', linewidth = 1)
		ax.set_xlabel('x position (mm)')
		ax.set_ylabel('y position (mm)')
		ax.set_zlabel('z position (mm)')
		plt.title('Photon Tracks, Run ' + str(self.run_id))
		plt.show()
		save_tracks = np.array(save_tracks)
		# np.save('/workspace/results/testing/[INSERT FILE NAME HERE]', filtered_tracks)

	def photon_shooting_angle(self, num_tracks = None, detected_only = True, reflected_only = False, diffuse_only = False):
		if num_tracks == None:
			num_tracks = self.num_particles

		mask = np.ones(self.num_particles, dtype=bool)

		if detected_only:
			mask &= self.tallies['SURFACE_DETECT']

		if reflected_only:
			mask &= self.particle_histories['REFLECT_SPECULAR'].astype(bool) 
				
		if diffuse_only:
			mask &= self.particle_histories['REFLECT_DIFFUSE'].astype(bool)

		filtered_tracks = self.photon_tracks[:, mask, :]
		print('shape of the filtered tracks',np.shape(filtered_tracks))

		x0 = filtered_tracks[0,:,0]
		y0 = filtered_tracks[0,:,1]
		z0 = filtered_tracks[0,:,2]
		x1 = filtered_tracks[1,:,0]
		y1 = filtered_tracks[1,:,1]
		z1 = filtered_tracks[1,:,2]
		angle = np.arccos(np.fabs(y1-y0)/np.sqrt(((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)))*(180./np.pi)
		fig = plt.figure()
		plt.hist(angle, bins = [theta for theta in range(91)])
		# hist is the height of the historgram
		hist, bin_edges = np.histogram(angle, bins = [theta for theta in range(91)])
		print(hist, bin_edges)

		plt.ylabel('Counts')
		plt.xlabel('Shooting Angle [deg]')
		plt.title('Emission Angle Distribution, Run ' + str(self.run_id))
		plt.tight_layout()
		plt.show()

		# the following is meant to reproduce the same incident histogram
		# print('number of steps to reach the SiPM',len(filtered_tracks)) #len(filtered_tracks) is the number of steps
		# last_x = []
		# last_y = []
		# last_z = []
		# second_last_x = []
		# second_last_y = []
		# second_last_z = []
		# for j in range(np.shape(filtered_tracks)[1]):
		# 	for i in range(len(filtered_tracks)):
		# 		# print(filtered_tracks[i,0,:])
		# 		if (filtered_tracks[i,j,0] == filtered_tracks[i+1,j,0]) and (filtered_tracks[i,j,1] == filtered_tracks[i+1,j,1]) and (filtered_tracks[i,j,2] == filtered_tracks[i+1,j,2]):
		# 			# print('the step where the position does not change is',i)
		# 			last_x.append(filtered_tracks[i,j,0])
		# 			second_last_x.append(filtered_tracks[i-1,j,0])
		# 			last_y.append(filtered_tracks[i,j,1])
		# 			second_last_y.append(filtered_tracks[i-1,j,1])
		# 			last_z.append(filtered_tracks[i,j,2])
		# 			second_last_z.append(filtered_tracks[i-1,j,2])
		# 			break	
		# # print(np.shape(last_z))
		# angle_list = []
		# for i in range(len(last_z)):
		# 	angle = np.arccos(np.fabs(last_y[i]-second_last_y[i])/np.sqrt(((last_x[i]-second_last_x[i])**2 + (last_y[i]-second_last_y[i])**2 + (last_z[i]-second_last_z[i])**2)))*(180./np.pi)
		# 	angle_list.append(angle)

		# fig = plt.figure()
		# plt.hist(angle_list, bins = [theta for theta in range(91)])
		# # hist is the height of the historgram
		# hist, bin_edges = np.histogram(angle_list, bins = [theta for theta in range(91)])
		# print(hist, bin_edges)

		# plt.ylabel('Counts')
		# plt.xlabel('Incident Angle [deg]')
		# plt.title('Incident Angle Distribution, Run ' + str(self.run_id))
		# plt.tight_layout()
		# plt.show()

#Sili: added on 02/07/2023 to plot the shooting angle and emission angle correlation of detected photons
	def photon_incident_angle_emission_angle_correlation(self, num_tracks = None, detected_only = True, reflected_specular_only = True, reflected_diffuse_only = False):
		if num_tracks == None:
			num_tracks = self.num_particles

		mask = np.ones(self.num_particles, dtype=bool)

		if detected_only:
			mask &= self.tallies['SURFACE_DETECT']

		if reflected_specular_only:
			mask &= self.particle_histories['REFLECT_SPECULAR'].astype(bool) 
			# print('reflect_specular_only')
				
		if reflected_diffuse_only:
			mask &= self.particle_histories['REFLECT_DIFFUSE'].astype(bool)

		filtered_tracks = self.photon_tracks[:, mask, :]
		print('shape of the filtered tracks',np.shape(filtered_tracks))
		x0 = filtered_tracks[0,:,0]
		y0 = filtered_tracks[0,:,1]
		z0 = filtered_tracks[0,:,2]
		x1 = filtered_tracks[1,:,0]
		y1 = filtered_tracks[1,:,1]
		z1 = filtered_tracks[1,:,2]
		emit_angle = np.arccos(np.fabs(y1-y0)/np.sqrt(((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)))*(180./np.pi)
		print('emit angle',emit_angle)
		# i = 1
		# while set(filtered_tracks[-i,:,0]) == set(filtered_tracks[-(i+1),:,0]):
		# 	i = i+1
		# 	print(i)

		last_x = []
		last_y = []
		last_z = []
		second_last_x = []
		second_last_y = []
		second_last_z = []
		for j in range(np.shape(filtered_tracks)[1]):
			for i in range(len(filtered_tracks)):
				# print(filtered_tracks[i,0,:])
				if (filtered_tracks[i,j,0] == filtered_tracks[i+1,j,0]) and (filtered_tracks[i,j,1] == filtered_tracks[i+1,j,1]) and (filtered_tracks[i,j,2] == filtered_tracks[i+1,j,2]):
					# print('the step where the position does not change is',i)
					last_x.append(filtered_tracks[i,j,0])
					second_last_x.append(filtered_tracks[i-1,j,0])
					last_y.append(filtered_tracks[i,j,1])
					second_last_y.append(filtered_tracks[i-1,j,1])
					last_z.append(filtered_tracks[i,j,2])
					second_last_z.append(filtered_tracks[i-1,j,2])
					break	

		inci_angle = []
		for i in range(len(last_z)):
			angle = np.arccos(np.fabs(last_y[i]-second_last_y[i])/np.sqrt(((last_x[i]-second_last_x[i])**2 + (last_y[i]-second_last_y[i])**2 + (last_z[i]-second_last_z[i])**2)))*(180./np.pi)
			inci_angle.append(angle)
		print('length of incident angle',len(inci_angle))

		plt.figure()
		# below is the colorbar histgram
		plt.hist2d(inci_angle,emit_angle,bins = [90,90],cmap = 'RdYlGn_r',norm = colors.LogNorm())
		plt.title('Incident vs. Emission angle of Detected & Reflected Photons')	
		# De_no,inci,emit = np.histogram2d(inci_angle,emit_angle,bins = [90,90])
		# print('De_no',np.shape(De_no))


		plt.xlabel('Incident angle (deg)')
		plt.ylabel('Emission angle (deg)')
		# plt.colorbar(boundaries = [-10] + [5], extend='both')
		# plt.colorbar()

		# #below is the scatter plot
		# plt.figure()
		# plt.scatter(inci_angle,emit_angle)
		# plt.xlabel('Incident angle (deg)')
		# plt.ylabel('Emission angle (deg)')
		# plt.title('Incident vs. Emission angle of Detected Reflected Photons')
		plt.xlim(0,90)
		plt.ylim(0,90)		
		plt.show()


	def plot_undetected_tracks(self,num_tracks = None):
		ax = plt.axes(projection = '3d')
		mask = np.zeros(self.num_particles, dtype = bool)
		for i in range(self.num_particles):
			if self.tallies['SURFACE_DETECT'][i] == False:
				mask[i] = True

		ax = plt.axes(projection = '3d')
		filtered_tracks = self.photon_tracks[:, mask, :]
		# print(np.shape(filtered_tracks))
		num_filtered_photon = len(filtered_tracks[1])
		max_tracks_to_plot = 200
		if num_filtered_photon > max_tracks_to_plot:
			num_plot_tracks = max_tracks_to_plot
		else:
			num_plot_tracks = num_filtered_photon
		save_tracks = []
		for j in range(num_plot_tracks):
			track = filtered_tracks[:,j,:]
			unique_ind = track[:-1, :] != track[1:, :]
			unique_ind = list(np.sum(unique_ind, axis = 1) > 0)
			unique_ind = [True] + unique_ind
			track = track[unique_ind, :]
			save_tracks.append(track)
			ax.plot(filtered_tracks[:, j, 0], filtered_tracks[:, j, 1], filtered_tracks[:, j, 2], color = 'green', linewidth = 1)
		ax.set_xlabel('x position (mm)')
		ax.set_ylabel('y position (mm)')
		ax.set_zlabel('z position (mm)')
		plt.title('Undetected Photon Tracks, Run ' + str(self.run_id))
		plt.show()
		np.save('/workspace/results/testing/hd3_filtered_undetected_tracks_test_' + str(self.run_id),save_tracks)


	#This function gives the right plot but doesn't save the right number of tracks. Not gonna use it; 
	def plot_tracks(self, num_tracks, detected_only = True, reflected_only = False, diffuse_only = False):
		if num_tracks == None:
			num_tracks = self.num_particles
			print(num_tracks)

		fig = plt.figure()
		ax = plt.axes(projection = '3d')
		all_tracks = []
		detected_tracks = [] #list of inds of detected photons
		reflected_tracks = [] #list of inds of reflected photons
		detected_reflected_tracks = []
		filtered_tracks = []
		# filter out all detected tracks, reflected track and reflected and detected/reflected tracks and save them to a file
		for idx in range(num_tracks):
			curr_positions = self.photon_tracks[:, idx, :] #select track of photon idx
			# ax.plot(curr_positions[:, 0], curr_positions[:, 1], curr_positions[:, 2], color = 'tab:blue', linewidth = 1)
			unique_ind = curr_positions[:-1, :] != curr_positions[1:, :] 
			unique_ind = list(np.sum(unique_ind, axis = 1) > 0)
			unique_ind = [True] + unique_ind
			curr_positions = curr_positions[unique_ind]
			all_tracks.append(curr_positions)
			did_detect = self.tallies['SURFACE_DETECT'][idx]
			did_reflect = self.particle_histories['REFLECT_SPECULAR'][idx].astype(bool) or self.particle_histories['REFLECT_DIFFUSE'][idx].astype(bool)
			if did_detect:
				detected_tracks.append(curr_positions)
			if did_reflect:
				reflected_tracks.append(curr_positions)
			if did_detect and did_reflect:
				detected_reflected_tracks.append(curr_positions)
			if detected_only:
				if did_detect:
					if reflected_only:
						if did_reflect:
							ax.plot(curr_positions[:, 0], curr_positions[:, 1], curr_positions[:, 2], color = 'tab:blue', linewidth = 1)
					else:
						ax.plot(curr_positions[:, 0], curr_positions[:, 1], curr_positions[:, 2], color = 'tab:blue', linewidth = 1)
			else:
				if reflected_only:
					if did_reflect:
						ax.plot(curr_positions[:, 0], curr_positions[:, 1], curr_positions[:, 2], color = 'tab:blue', linewidth = 1)
				else:
					ax.plot(curr_positions[:, 0], curr_positions[:, 1], curr_positions[:, 2], color = 'tab:blue', linewidth = 1)
		print('the dimension for all tracks',np.shape(all_tracks))
		print('the dimension for detected tracks',np.shape(detected_tracks))
		print('the dimension for reflected tracks',np.shape(reflected_tracks))
		print('the dimension for detected and reflected tracks',np.shape(detected_reflected_tracks))			
		ax.set_xlabel('x position (mm)')
		ax.set_ylabel('y position (mm)')
		ax.set_zlabel('z position (mm)')
		plt.title('Photon Tracks, Run ' + str(self.run_id))
		plt.show()
		# print(self.photon_tracks[0,:,1]) #print the initial position of all tracks
		# np.save('/workspace/results/testing/hd3_tracks_test_' + str(self.run_id),all_tracks)
		np.save('/workspace/results/testing/hd3_detrefl_tracks_test_' + str(self.run_id),detected_reflected_tracks)
		np.save('/workspace/results/testing/hd3_det_tracks_test_' + str(self.run_id),detected_tracks)

	def plot_angle_hist(self,histogramfilename):
		lxe_refractive_index = self.gm.mat_manager.material_props['liquid xenon']['refractive_index']
		fig = plt.figure()
		plt.hist(self.detected_angles, bins = [theta for theta in range(91)])
		# hist is the height of the historgram
		hist, bin_edges = np.histogram(self.detected_angles, bins = [theta for theta in range(91)])
		print(hist)

		save_data = {
				'Incident Angle': np.array(list(range(0,90))),
				'Detected number': hist,
				'LXe refractive index':np.array([lxe_refractive_index for _ in range(len(hist))])}
		df = pd.DataFrame(save_data)
		df.to_csv(histogramfilename)

		# # anglecount = 0
		# # for i in range(len(self.detected_angles)):
		# # 	if self.detected_angles[i] >= 32 and self.detected_angles[i] <= 33:
		# # 		anglecount = anglecount + 1
		# # print(f"the number of 32 degree incidences is {anglecount}")
		plt.ylabel('Counts')
		plt.xlabel('Incident Angle [deg]')
		plt.title('Incident Angle Distribution, Run ' + str(self.run_id))
		plt.tight_layout()
		plt.show()

	def plot_position_hist(self):
		fig = plt.figure()
		plt.hist2d(
			self.detected_positions[:, 0],
			self.detected_positions[:, 2],
			bins = [np.linspace(85, 97.5, 50), np.linspace(85, 97.5, 50)])
		cbar = plt.colorbar()
		cbar.set_label('Counts')
		plt.xlabel('x position (mm)')
		plt.ylabel('z position (mm)')
		plt.title('Position Distribution, Run ' + str(self.run_id))
		plt.tight_layout()
		plt.show()

	def plot_refl_multiplicity(self, do_log = True, density = True):
		bins = [x for x in range(10)]
		spec_reflection_data = self.particle_histories['REFLECT_SPECULAR']
		# spec_reflection_data = spec_reflection_data[spec_reflection_data > 0]
		spec_reflection_data_det = spec_reflection_data[self.tallies['SURFACE_DETECT']]
		
		# plt.hist(spec_reflection_data, bins = bins, density = density, label = 'All Photons', histtype = 'step')
		plt.hist(spec_reflection_data_det, bins = bins, density = density, label = 'Det. Photons', histtype = 'step')
		plt.xlabel('Number of Reflections')
		plt.ylabel('Fraction of Photons')
		plt.title('Reflection Multiplicity')
		plt.legend()
		if do_log:
			plt.yscale('log')
		plt.tight_layout()
		plt.show()

		
	#adding filter for specific number of reflections
	def plot_filtered_tracks(self, filter_by, num_tracks = None):

		mask = np.zeros(self.num_particles, dtype = bool)

		for i in range(self.num_particles):
			if self.particle_histories['REFLECT_SPECULAR'][i] == filter_by and self.tallies['SURFACE_DETECT'][i] == True:
				mask[i] = True
			elif self.particle_histories['REFLECT_DIFFUSE'][i] == filter_by and self.tallies['SURFACE_DETECT'][i] == True:
				mask[i] = True


		fig = plt.figure()
		ax = plt.axes(projection = '3d')
		filtered_tracks = self.photon_tracks[:, mask, :]
		print("number of filtered tracks:", np.shape(filtered_tracks))

		if num_tracks > len(filtered_tracks[0]):
			num_tracks = len(filtered_tracks[0])

		for j in range(num_tracks):
			ax.plot(filtered_tracks[:, j, 0], filtered_tracks[:, j, 1], filtered_tracks[:, j, 2], color = 'tab:blue', linewidth = 1)

		ax.set_xlabel('x position (mm)')
		ax.set_ylabel('y position (mm)')
		ax.set_zlabel('z position (mm)')
		plt.title('Filtered Photon Tracks from Reflection, Run ' + str(self.run_id))
		plt.show()

		np.save('/workspace/results/testing/hd3_filtered_tracks_test_' + str(self.run_id),filtered_tracks)

	#Filter out path with only scattering without reflection. Intended to study why there is no scattering from angle 7 to 21 and 60 to 67 degree
	def plot_filtered_scattered_tracks(self, num_tracks = None):
		mask = np.zeros(self.num_particles, dtype = bool)
		for i in range(self.num_particles):
			if self.particle_histories['RAYLEIGH_SCATTER'][i] != 0 and self.tallies['SURFACE_DETECT'][i] == False and self.particle_histories['REFLECT_SPECULAR'][i] == 0:
				# print(self.particle_histories['RAYLEIGH_SCATTER'][i])
				mask[i] = True


		fig = plt.figure()
		ax = plt.axes(projection = '3d')
		filtered_tracks = self.photon_tracks[:, mask, :]		
		print(len(filtered_tracks[1])) #number of photons that is scattered/detected 
		# print(filtered_tracks[1])
		# print(len(filtered_tracks)) #number of steps that a single photon propogate, usually the steps it take until being detected is much less than this number
		# print(filtered_tracks.shape)
		if num_tracks > len(filtered_tracks):
			num_tracks = len(filtered_tracks)
		# print(num_tracks)

		# for j in range(2):
		# 	plt.plot(filtered_tracks[j, 0, 0], filtered_tracks[j, 0, 1], filtered_tracks[j, 0, 2], 'o' ,color = 'green')
		
		#just select 100 photons to plot their tracks
		for j in range(100):
			ax.plot(filtered_tracks[:, j, 0], filtered_tracks[:, j, 1], filtered_tracks[:, j, 2], color = 'green', linewidth = 1)
		# 	print(filtered_tracks[:, j, 0]) #for jth scatter/detected photon at all step, print the x position
		ax.set_xlabel('x position (mm)')
		ax.set_ylabel('y position (mm)')
		ax.set_zlabel('z position (mm)')
		plt.title('Undetected Filtered Photon Tracks from just Scattering, Run ' + str(self.run_id))
		plt.show()

		np.save('/workspace/results/testing/hd3_filtered_scatter_tracks_test_' + str(self.run_id),filtered_tracks)

	def plot_refl_angle(self, do_log = True, low_angle = 0, high_angle = 91):
		bins_refl = [x for x in range(10)]
		bins_angle = [x for x in range(low_angle, high_angle)]
		spec_reflection_data = self.particle_histories['REFLECT_SPECULAR']
		spec_reflection_data_det = spec_reflection_data[self.tallies['SURFACE_DETECT']]
		# print(bins_refl)
		# print(bins_angle)
		# print([bins_angle, bins_refl])

		plt.figure()
		if do_log:
			plt.hist2d(self.detected_angles, spec_reflection_data_det, bins = [bins_angle, bins_refl], norm=mpl.colors.LogNorm())
		else:
			plt.hist2d(self.detected_angles, spec_reflection_data_det, bins = [bins_angle, bins_refl])

		print(self.detected_angles, spec_reflection_data_det)
		plt.xlabel('Incident Angle (deg)')
		plt.ylabel('Reflection Multiplicity')
		plt.colorbar()
		plt.tight_layout()
		plt.show()







