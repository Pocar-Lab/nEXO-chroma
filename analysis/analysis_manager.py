#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from stl import mesh
from matplotlib import colors
from mpl_toolkits import mplot3d
from array import array


class analysis_manager:
	"""
    Manages the analysis of photon tracks and generates various plots.

    Attributes
    ----------
    gm : object
        Instance of the geometry_manager class.
    experiment_name : str
        String used to identify each experiment.
    photons : ndarray
        Array of photons.
    photon_tracks : int
        Number of photon tracks.
    run_id : int
        ID of the run.
    seed : int
        Seed for random number generation.
    particle_histories : dict
        Dictionary containing particle histories.
    selected_plots : list
        List of selected plots to generate.
    all_tracks : list
        List to store all photon tracks.
    detected_tracks : list
        List to store detected photon tracks.
    undetected_tracks : list
        List to store undetected photon tracks.
    reflected_tracks : list
        List to store reflected photon tracks.
    filtered_scattered_tracks : list
        List to store scattered but not detected or specularly reflected photon tracks.
    detected_reflected_tracks : list
        List to store detected and reflected photon tracks.
    specular_reflected_tracks : list
        List to store specularly reflected photon tracks.
    diffuse_reflected_tracks : list
        List to store diffusely reflected photon tracks.
    num_particles : int
        Number of particles.
    num_tracks : int
        Number of tracks.
    plots : list
        List of plots.
    geometry_data_path : str
        Path to the geometry data file.
    plot_functions : dict
        Dictionary of plot functions.
    tallies : dict
        Dictionary to store tallies.
    efficiency : float
        Efficiency of photon detection.
    detected_positions : ndarray
        Positions of detected photons.
    detected_angles : ndarray
        Angles of detected photons.
    emit_angle : int
        Emission angle.
    """
	def __init__(self, geometry_manager, experiment_name, selected_plots, photons, photon_tracks = 1000, run_id = 0, seed = 0, histories = None):
		"""
        Initializes the analysis manager.

        Parameters
        ----------
        geometry_manager : object
            Geometry manager.
        experiment_name : str
            Name of the experiment.
        selected_plots : list
            List of selected plots to generate.
        photons : ndarray
            Array of photons.
        photon_tracks : int, optional
            Number of photon tracks (default is 1000).
        run_id : int, optional
            ID of the run (default is 0).
        seed : int, optional
            Seed for random number generation (default is 0).
        histories : dict, optional
            Dictionary containing particle histories (default is None).
        """
		self.gm = geometry_manager
		self.experiment_name = experiment_name
		self.photons = photons
		self.photon_tracks = photon_tracks
		self.all_tracks = None
		self.detected_tracks = None
		self.undetected_tracks = None
		self.reflected_tracks = None
		self.filtered_scattered_tracks = None
		self.detected_reflected_tracks = None
		self.specular_reflected_tracks = None
		self.diffuse_reflected_tracks = None	
		self.num_particles = len(self.photons)
		self.run_id = run_id
		self.seed = seed
		self.particle_histories = histories
		self.selected_plots = selected_plots
		self.get_tallies()
		self.num_tracks = 100
		self.plots = selected_plots
		self.geometry_data_path = f'/workspace/data_files/data/{self.experiment_name}/geometry_components_{self.experiment_name}.csv'


		self.plot_functions = {
			'plot_all_tracks' : self.plot_all_tracks_wrapper,
			'plot_detected_tracks' : self.plot_detected_tracks_wrapper,
			'plot_undetected_tracks' : self.plot_undetected_tracks_wrapper,
			'plot_reflected_tracks' : self.plot_reflected_tracks_wrapper,
			'plot_filtered_scattered_tracks' : self.plot_filtered_scattered_tracks_wrapper,
			'plot_detected_reflected_tracks' : self.plot_detected_reflected_tracks_wrapper,
			'plot_specular_reflected_tracks' : self.plot_specular_reflected_tracks_wrapper,
			'plot_diffuse_reflected_tracks' : self.plot_diffuse_reflected_tracks_wrapper,
            'plot_refl_multiplicity': self.plot_refl_multiplicity_wrapper,
            'photon_shooting_angle': self.photon_shooting_angle_wrapper,
            'photon_incident_angle_emission_angle_correlation': self.photon_incident_angle_emission_angle_correlation_wrapper,
            'plot_angle_hist': self.plot_angle_hist_wrapper,
            'plot_refl_angle': self.plot_refl_angle_wrapper,
            'plot_position_hist': self.plot_position_hist_wrapper
        }

		
		histogram_file_name = f'/workspace/results/{self.experiment_name}/histogram_data_seed_{self.seed}'
		filename = f'/workspace/results/{self.experiment_name}/datapoints/hd3_data_test_seed_{self.seed}.csv'

		self.preprocess_tracks()
		self.execute_plots()
		self.save_detected(filename)


	def preprocess_tracks(self):
		"""
        Preprocesses the photon tracks and categorizes them.
        """
		num_particles = self.num_particles
		self.all_tracks = []
		self.detected_tracks = []     			# tracks of photons detected
		self.undetected_tracks = []				# tracks of photons not detected
		self.reflected_tracks = []				# tracks of photons reflected
		self.filtered_scattered_tracks = []		# tracks of photons scattered but not detected or speculrly reflected
		self.detected_reflected_tracks = []		# tracks of photons both detected and reflected
		self.specular_reflected_tracks = []		# tracks of photons specularly reflected
		self.diffuse_reflected_tracks = []		# tracks of photons diffusively reflected

		for idx in range(num_particles):
			curr_positions = self.photon_tracks[:, idx, :]  # select track of photon idx

			# Remove duplicate consecutive positions
			unique_ind = curr_positions[:-1, :] != curr_positions[1:, :]
			unique_ind = list(np.sum(unique_ind, axis=1) > 0)
			unique_ind = [True] + unique_ind

			curr_positions = curr_positions[unique_ind]
			self.all_tracks.append(curr_positions)

			did_detect = self.tallies['SURFACE_DETECT'][idx]
			did_reflect_specular = self.particle_histories['REFLECT_SPECULAR'][idx].astype(bool)
			did_reflect_diffuse = self.particle_histories['REFLECT_DIFFUSE'][idx].astype(bool)
			did_scatter = self.particle_histories['RAYLEIGH_SCATTER'][idx] != 0

			if did_detect:
				self.detected_tracks.append(curr_positions)
			else:
				self.undetected_tracks.append(curr_positions)

			if did_reflect_specular:
				self.specular_reflected_tracks.append(curr_positions)
				self.reflected_tracks.append(curr_positions)
			if did_reflect_diffuse:
				self.diffuse_reflected_tracks.append(curr_positions)
				self.reflected_tracks.append(curr_positions)

			if did_detect and (did_reflect_specular or did_reflect_diffuse):
				self.detected_reflected_tracks.append(curr_positions)
			if did_scatter and not did_detect and not did_reflect_specular:
				self.filtered_scattered_tracks.append(curr_positions)


	def plot_tracks(self, tracks, title, plot_geometry, color='tab:blue', linewidth=1):
		"""
        Plots the photon tracks in 3D.

        Parameters
        ----------
        tracks : list
            List of photon tracks to plot.
        title : str
            Title of the plot.
        plot_geometry : bool
            Whether to plot the geometry.
        color : str, optional
            Color of the tracks (default is 'tab:blue').
        linewidth : int, optional
            Line width of the tracks (default is 1).
        """

		figure = plt.figure()
		axes = mplot3d.Axes3D(figure)

		num_tracks = self.num_tracks if self.num_tracks < len(tracks) else len(tracks)
		for i in range(num_tracks):
			track = tracks[i]
			#ax.plot(track[:, 0], track[:, 1], track[:, 2], color=color, linewidth=linewidth)
			axes.plot(track[:, 0], track[:, 1], track[:, 2], color=color, linewidth=linewidth)

		if plot_geometry:
			geometry_df = pd.read_csv(self.geometry_data_path)
			stl_names = geometry_df['stl_filepath']
			colors = geometry_df['color']
			y_displacement = geometry_df['displacement y']
			z_displacement = geometry_df['displacement z']

			for curr_filename, curr_color, current_y_displacement, current_z_displacement in zip(stl_names, colors, y_displacement, z_displacement):
				your_mesh = mesh.Mesh.from_file(curr_filename)
				mesh_dimension = np.shape(your_mesh.vectors)
				for i in range(mesh_dimension[1]):
					your_mesh.vectors[:,i,1] += current_y_displacement
				for k in range(mesh_dimension[1]):
					your_mesh.vectors[:,i,2] += current_z_displacement
				poly3d = mplot3d.art3d.Poly3DCollection(your_mesh.vectors)
				poly3d.set_alpha(0.2)
				poly3d.set_edgecolor(None)
				poly3d.set_facecolor(curr_color)
				axes.add_collection3d(poly3d)
			# Auto scale to the mesh size
			scale = your_mesh.points.flatten()

			axes.auto_scale_xyz(scale, scale, scale)
		axes.set_xlabel('x position (mm)')
		axes.set_ylabel('y position (mm)')
		axes.set_zlabel('z position (mm)')
		figure.suptitle(title)
		plt.show()

	def incident_angle(self, last_pos):
		"""
        Calculates the incident angle of photons.

        Parameters
        ----------
        last_pos : ndarray
            Array of last positions of the photons.

        Returns
        -------
        ndarray
            Array of incident angles in degrees.
        """
		angles = np.arccos(np.fabs(last_pos[:,1])/np.sqrt((last_pos[:,0]**2 + last_pos[:,1]**2 + last_pos[:,2]**2)))*(180./np.pi)
		return angles

	def get_tallies(self):
		"""
        Retrieves and prints tallies of different photon interactions.
        """
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
		"""
        Saves the detected photon data to a CSV file.

        Parameters
        ----------
        filename : str
            Path to the file where the data will be saved.
        """
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
		df = pd.DataFrame(save_data)
		df.to_csv(filename)


	def photon_shooting_angle(self, num_tracks = None, detected_only = True, reflected_only = False, diffuse_only = False):
		"""
        Plots the distribution of photon shooting angles.

        Parameters
        ----------
        num_tracks : int, optional
            Number of tracks to plot (default is None, which means all photons will be plotted.).
        detected_only : bool, optional
            Whether to include only detected photons (default is True).
        reflected_only : bool, optional
            Whether to include only reflected photons (default is False).
        diffuse_only : bool, optional
            Whether to include only diffusely reflected photons (default is False).
        """
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
	

#Sili: added on 02/07/2023 to plot the shooting angle and emission angle correlation of detected photons
	def photon_incident_angle_emission_angle_correlation(self, num_tracks = None, detected_only = True, reflected_specular_only = True, reflected_diffuse_only = False):
		"""
        Plots the correlation between photon incident and emission angles.

        Parameters
        ----------
        num_tracks : int, optional
            Number of tracks to plot (default is None, which means all).
        detected_only : bool, optional
            Whether to include only detected photons (default is True).
        reflected_specular_only : bool, optional
            Whether to include only specularly reflected photons (default is True).
        reflected_diffuse_only : bool, optional
            Whether to include only diffusely reflected photons (default is False).
        """
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

	def plot_angle_hist(self,histogramfilename, showPlot = True):
		"""
        Plots a histogram of the detected photon angles and saves the data to a CSV file.

        Parameters
        ----------
        histogramfilename : str
            Path to the file where the histogram data will be saved.
        showPlot : bool, optional
            Whether to show the plot (default is True).

        Returns
        -------
        ndarray
            Array of histogram values.
        """
     
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

		plt.ylabel('Counts')
		plt.xlabel('Incident Angle [deg]')
		plt.title('Incident Angle Distribution, Run ' + str(self.run_id))
		plt.tight_layout()
		if showPlot:
			plt.show()
		return hist

	def plot_position_hist(self):
		"""
        Plots a 2D histogram of the detected photon positions.
        """
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
		"""
        Plots the reflection multiplicity of photons.

        Parameters
        ----------
        do_log : bool, optional
            Whether to use a logarithmic scale for the y-axis (default is True).
        density : bool, optional
            Whether to normalize the histogram (default is True).
        """
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

	def plot_refl_angle(self, do_log = True, low_angle = 0, high_angle = 91):
		"""
        Plots the reflection angle of detected photons.

        Parameters
        ----------
        do_log : bool, optional
            Whether to use a logarithmic scale for the color bar (default is True).
        low_angle : int, optional
            Lower bound for the angle histogram (default is 0).
        high_angle : int, optional
            Upper bound for the angle histogram (default is 91).
        """
		bins_refl = [x for x in range(10)]
		bins_angle = [x for x in range(low_angle, high_angle)]
		spec_reflection_data = self.particle_histories['REFLECT_SPECULAR']
		spec_reflection_data_det = spec_reflection_data[self.tallies['SURFACE_DETECT']]

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

	def plot_all_tracks_wrapper(self):
		title = f'Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.all_tracks, title, False)

	def plot_detected_tracks_wrapper(self):
		title = f'Detected Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.detected_tracks, title, False)

	def plot_undetected_tracks_wrapper(self):
		title = f' Undetected Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.undetected_tracks, title, False)

	def plot_reflected_tracks_wrapper(self):
		title = f'Reflected Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.reflected_tracks, title, False)

	def plot_filtered_scattered_tracks_wrapper(self):
		title = f'Filtered Scattered Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.filtered_scattered_tracks, title, False)

	def plot_detected_reflected_tracks_wrapper(self):
		title = f'Detected and Reflected Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.detected_reflected_tracks, title, False)

	def plot_specular_reflected_tracks_wrapper(self):
		title = f'Specularly Refelcted Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.specular_reflected_tracks, title, False)

	def plot_diffuse_reflected_tracks_wrapper(self):
		title = f' Diffusively Reflected Photon Tracks, Seed {self.seed}, Run {self.run_id}'
		self.plot_tracks(self.diffuse_reflected_tracks, title, False)

	def plot_refl_multiplicity_wrapper(self):
		self.plot_refl_multiplicity(density=True)

	def photon_shooting_angle_wrapper(self):
		self.photon_shooting_angle(num_tracks=None, detected_only=True, reflected_only=False)

	def photon_incident_angle_emission_angle_correlation_wrapper(self):
		self.photon_incident_angle_emission_angle_correlation(num_tracks=None, detected_only=True, reflected_specular_only=False, reflected_diffuse_only=False)

	def plot_angle_hist_wrapper(self, showPlot = True):
		histogram_file_name = f'/workspace/results/{self.experiment_name}/histogram_seed:{self.seed}'
		return self.plot_angle_hist(histogram_file_name, showPlot)

	def plot_refl_angle_wrapper(self):
		self.plot_refl_angle(low_angle=12, do_log=False)

	def plot_position_hist_wrapper(self):
		self.plot_position_hist()

	def execute_plots(self):
		"""
        Executes the selected plot functions defined in self.plot_functions.
        """
		for plot_name in self.plots:
			if plot_name in self.plot_functions:
				self.plot_functions[plot_name]()
			else:
				print(f"Plot '{plot_name}' is not recognized. Available plots are: {', '.join(self.plot_functions.keys())}")





