#!/usr/bin/env python
from chroma.detector import Detector
from chroma.stl import mesh_from_stl
from chroma.geometry import Solid
from chroma import view
from chroma.loader import load_bvh


import pandas as pd
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from material_manager import material_manager
from surface_manager import surface_manager
# all components, general geometry, manage surface and material
class geometry_manager:
    def __init__(self, experiment_name, run_id,visualize = False):

        self.experiment_name = experiment_name
        self.mat_manager = material_manager(self.experiment_name, run_id)
        self.surf_manager = surface_manager(self.mat_manager, self.experiment_name, run_id)
        self.run_id = run_id
        # print("run_id in gm init",run_id)
        # define chroma simulation
        self.global_geometry = Detector(self.mat_manager.global_material)
        #A Detector is a subclass of Geometry that allows some Solids to be marked as photon detectors, which we will suggestively call "PMTs
        # self.geometry_data_path = '/workspace/data_files/geometry_components.csv'
        # self.geometry_data_path = '/workspace/data_files/data/copperplates_06.23.2022/geometry_components _copper_plate.csv'
        # # starting from 06/30/2022, try the beam direction by moving the source displacement
        # self.geometry_data_path = '/workspace/data_files/data/beam_direction_06.30.2022/geometry_components _copper_plate0630.csv'     
        # self.geometry_data_path = '/workspace/data_files/data/silica_window_07.18.2022/geometry_components _silica_window_07182022.csv'        
        # self.geometry_data_path = '/workspace/data_files/data/source_copperholder_08.16.2022/geometry_components _sourceCu_holder_08162022.csv'
        # self.geometry_data_path = '/workspace/data_files/data/copper_gasket_08.29.2022/geometry_components _coppergasket_08292022.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Al_filler_02.07.2023/geometry_components __Alfiller_02072023.csv'
        # self.geometry_data_path = '/workspace/data_files/data/sourcepart_05.11.2023/geometry_components __sourcepart_05112023.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_woteflon_05.12.2023/geometry_components __Sebastian_woteflon_0512.csv.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_teflon_05.12 .2023/geometry_components __Sebastian_teflon_0512.csv'
        # self.geometry_data_path ='/workspace/data_files/data/Sebastian_teflon_05.23.2023/geometry_components __Sebastian_teflon_0523.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_woteflon_05.23.2023/geometry_components __Sebastian_woteflon_0523.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_woteflon_upper_06.05.2023/geometry_components __Sebastian_woteflon_upper_0605.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_teflon_upperlimit_06.05.2023/geometry_components __Sebastian_teflon__upper_0605.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_teflon_lowerlimit_06.05.2023/geometry_components __Sebastian_teflon__lower_0605.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_flippedsource_06.06.2023/geometry_components __Sebastian_teflon__FS_0606.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_woteflon_geom_07.22.2022/geometry_components __Sebastian_woteflon_0722_2022.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_FS_06.08.2023_correctedSiPM/geometry_components __Sebastian_teflon__FS_0608_corrected.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_03.31.2023(liquefaction_corrrectedSiPM)/geometry_components __Sebastian_03312023_corrected.csv'
        # self.geometry_data_path ='/workspace/data_files/data/Sebastian_05.18.2023(liquefaction)_correctedSiPM/geometry_components __Sebastian_teflon__0518_corrected.csv'
        # self.geometry_data_path ='/workspace/data_files/data/Sebastian_07.20.2022(liquefaction)_corrrectedSiPM/geometry_components __Sebastian_07202022_corrected.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_08.01.2023(liquefaction)_correctedSiPM/geometry_components __Sebastian_Pdreflector__0801_corrected.csv'
        # self.geometry_data_path = '/workspace/data_files/data/Sebastian_FS_06.08.2023_correctedSiPM/geometry_components __Sebastian_teflon__FS_0608_corrected (FusionTest).csv'
        self.geometry_data_path = '/workspace/data_files/data/'+ experiment_name + '/geometry_components_' + experiment_name + '.csv'
        self.build_geometry()

        self.global_geometry.flatten()
        if visualize:
            view(self.global_geometry)
        self.global_geometry.bvh = load_bvh(self.global_geometry)

    def build_geometry(self):
        #print('run_id in build_geometry',self.run_id)
        # read in the csv file into dataframe
        self.geometry_df = pd.read_csv(self.geometry_data_path)

        # iterate through all geometries and create Solid object, store into dictionary of solids
        self.solids = {}
        for index, row in self.geometry_df.iterrows():
            curr_name = row['name']

            mesh = mesh_from_stl(filename = row['stl_filepath'])#convert the stl files to mesh used in Chroma?
            inner_mat = self.mat_manager.get_material(row['inner_mat'])
            outer_mat = self.mat_manager.get_material(row['outer_mat'])
            surface = self.surf_manager.get_surface(row['surface'])

            color = int(colors.cnames[row['color']][1:], 16)

            curr_displacement = (row['displacement x'], row['displacement y'], row['displacement z'])
            print(curr_displacement)       
                
            # #below shift the vertical displacement
            # if curr_name == "silica gasket 1" or curr_name == "silica gasket 2" or curr_name == "silica window":
            #     curr_displacement = (row['displacement x'], row['displacement y']+self.run_id*0.01, row['displacement z'])
            #     print(curr_displacement)   
            #     # print(type(row['displacement y']))        
            # #Solid object attaches materials, surfaces, and colors to each triangle in a Mesh object
            # else:
            #     curr_displacement = (row['displacement x'], row['displacement y'], row['displacement z'])
            #     # print(curr_displacement)

            self.solids[curr_name] =  Solid(mesh = mesh, 
                                            material1 = inner_mat,
                                            material2 = outer_mat,
                                            surface = surface,
                                            color = color)
            # check to see if it is a detecting volume, add to geometry, (pmt and other solid will be treated differently?)
            if row['solid_type'] == 'pmt':
                self.global_geometry.add_pmt(pmt = self.solids[curr_name],
                                            rotation = None,
                                            displacement = curr_displacement)
            elif row['solid_type'] == 'solid':
                self.global_geometry.add_solid(solid = self.solids[curr_name],
                                            rotation = None,
                                            displacement = curr_displacement)       

    def get_solid_center(self, name):
        #this solid center is to define the centre of the photon source? Use it later?
        curr_mesh_triangles = self.solids[name].mesh.get_triangle_centers()
        return [np.mean(curr_mesh_triangles[:, 0]),
                np.mean(curr_mesh_triangles[:, 1]),
                np.mean(curr_mesh_triangles[:, 2])]
