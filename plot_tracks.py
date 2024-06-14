
import pandas as pd
import numpy as np
#from chroma.stl import mesh_from_stl as mesh
#import stl.mesh as mesh
from chroma.stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Create a new plot, plot tracks
figure = plt.figure()
axes = mplot3d.Axes3D(figure)
# tracks = np.load('/workspace/results/testing/hd3_tracks_test_0.npy', allow_pickle = True)
# tracks = np.load('/workspace/results/testing/hd3_detrefl_tracks_test_0.npy', allow_pickle = True)
tracks = np.load('/workspace/results/testing/hd3_det_tracks_test_0.npy', allow_pickle = True)
# # tracks = np.load('/workspace/results/testing/hd3_filtered_scatter_tracks_test_0.npy', allow_pickle = True)

# tracks = np.load('/workspace/results/testing/hd3_filtered_tracks_test_0.npy', allow_pickle = True)
# tracks = np.load('/workspace/results/testing/hd3_filtered_undetected_tracks_test_0.npy', allow_pickle = True)
print(np.shape(tracks))
print(len(tracks))

# for i in range(len(tracks)):
# 	curr_positions = tracks[i]
# 	axes.plot(curr_positions[:, 0], curr_positions[:, 1], curr_positions[:, 2], color = 'k', linewidth = 1)

# #Reed's rewrite, sometimes work
# figure = plt.figure()
# axes = plt..axe(projection = '3d')
# tracks='track here'
# for i in range(len(tracks[0])):
# 	curr_positions=tracks
# 	axes.plot(curr_positions[:,i,0], curr_positions[:,i,1], curr_positions[:,1,2],color = 'k',linewidth = 1)

# Load the STL files and add the vectors to the plot

# geom_file = '/workspace/data_files/geometry_components.csv'
# geom_file = '/workspace/results/data/copperplates_06.23.2022/geometry_components _copper_plate.csv'
# geom_file = '/workspace/results/data/beam_direction_06.30.2022/geometry_components _copper_plate0630.csv'
# geom_file = '/workspace/results/data/silica_window_07.18.2022/geometry_components _silica_window_07182022.csv'
# geom_file = '/workspace/results/data/source_copperholder_08.16.2022/geometry_components _sourceCu_holder_08162022.csv'
# geom_file = '/workspace/results/data/copper_gasket_08.29.2022/geometry_components _coppergasket_08292022.csv'
# geom_file = '/workspace/results/data/Al_filler_02.07.2023/geometry_components __Alfiller_02072023.csv'
# geom_file = '/workspace/data_files/stl_files/Sebastian/STL 3/stl3.csv'
# geom_file = '/workspace/results/data/Sebastian_woteflon_05.12.2023/geometry_components __Sebastian_woteflon_0512.csv.csv'
# geom_file = '/workspace/results/data/Sebastian_teflon_05.12 .2023/geometry_components __Sebastian_teflon_0512.csv'
# geom_file ='/workspace/results/data/Sebastian_woteflon_05.23.2023/geometry_components __Sebastian_woteflon_0523.csv'
# geom_file ='/workspace/results/data/Sebastian_teflon_05.23.2023/geometry_components __Sebastian_teflon_0523.csv'
# geom_file = '/workspace/results/data/Sebastian_woteflon_upper_06.05.2023/geometry_components __Sebastian_woteflon_upper_0605.csv'
# geom_file = '/workspace/results/data/Sebastian_teflon_lowerlimit_06.05.2023/geometry_components __Sebastian_teflon__lower_0605.csv'
# geom_file = '/workspace/results/data/Sebastian_teflon_upperlimit_06.05.2023/geometry_components __Sebastian_teflon__upper_0605.csv'
# geom_file = '/workspace/results/data/Sebastian_flippedsource_06.06.2023/geometry_components __Sebastian_teflon__FS_0606.csv'
# geom_file = '/workspace/results/data/Sebastian_woteflon_geom_07.22.2022/geometry_components __Sebastian_woteflon_0722_2022.csv'
# geom_file = '/workspace/results/data/Sebastian_FS_06.08.2023_correctedSiPM/geometry_components __Sebastian_teflon__FS_0608_corrected.csv'
# geom_file ='/workspace/results/data/Sebastian_03.31.2023(liquefaction_corrrectedSiPM)/geometry_components __Sebastian_03312023_corrected.csv'
# geom_file ='/workspace/results/data/Sebastian_05.18.2023(liquefaction)_correctedSiPM/geometry_components __Sebastian_teflon__0518_corrected.csv'
# geom_file = '/workspace/results/data/Sebastian_07.20.2022(liquefaction)_corrrectedSiPM/geometry_components __Sebastian_07202022_corrected.csv'
geom_file = '/workspace/results/data/Sebastian_08.01.2023(liquefaction)_correctedSiPM/geometry_components __Sebastian_Pdreflector__0801_corrected.csv'
# geom_file = '/workspace/results/data/Sebastian_FS_06.08.2023_correctedSiPM/geometry_components __Sebastian_teflon__FS_0608_corrected (FusionTest).csv'
geometry_df = pd.read_csv(geom_file)
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

# # plot specific geometry
# # find dimension of specific geometry in the simulation
# curr_filename = '/workspace/data_files/stl_files/Sebastian/2023_6_20_Copper_teflon_corrected/CopperReflectorAssemwithFO - SipmSubAssembly-1 SipmPlate-1.STL'
# your_mesh = mesh.Mesh.from_file(curr_filename)
# # print(current_y_displacement)
# mesh_dimension = np.shape(your_mesh.vectors)
# print(mesh_dimension)
# # print('the y component of mesh:',your_mesh.vectors[:,:,1])

# print('the minimum y is:',your_mesh.vectors[:,:,1].min())
# print('the maximum y is:',your_mesh.vectors[:,:,1].max())
# print('the minimum x is:',your_mesh.vectors[:,:,0].min())
# print('the maximum x is:',your_mesh.vectors[:,:,0].max())
# print('the minimum z is:',your_mesh.vectors[:,:,2].min())
# print('the maximum z is:',your_mesh.vectors[:,:,2].max())
                                                                                                                    
# poly3d = mplot3d.art3d.Poly3DCollection(your_mesh.vectors)
# poly3d.set_alpha(0.2)
# poly3d.set_edgecolor(None)
# poly3d.set_facecolor('grey')
# axes.add_collection3d(poly3d)

# axes.set_xlabel('x [mm]')
# axes.set_ylabel('y [mm]')
# axes.set_zlabel('z [mm]')
axes.set_xlim(0, 60)
axes.set_ylim(30, 90)
axes.set_zlim(0, 130)

# axes.set_xlim(50, 90)
# axes.set_ylim(5, 40)
# axes.set_zlim(0, 60)
# # Show the plot to the screen
plt.show()
