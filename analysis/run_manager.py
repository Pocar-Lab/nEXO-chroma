#!/usr/bin/env python

from chroma.sim import Simulation
from chroma.event import Photons

import numpy as np
import math
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from chroma import gpu

from .analysis_manager import analysis_manager


class run_manager:
    """
    Manages the simulation run, including initializing the simulation, generating primary photons,
    and propagating photons through the geometry.

    :param geometry_manager: The geometry manager that handles the geometry of the simulation.
    :type geometry_manager: GeometryManager
    :param experiment_name: The name of the experiment.
    :type experiment_name: str
    :param random_seed: The random seed for reproducibility.
    :type random_seed: int
    :param num_particles: The number of particles to simulate.
    :type num_particles: int
    :param run_id: The unique identifier for the run.
    :type run_id: int
    :param plots: The plots to be generated during analysis.
    :type plots: list
    :type write: boolean
    """

    def __init__(
        self,
        geometry_manager,
        experiment_name,
        random_seed,
        num_particles,
        run_id,
        plots,
        write=False,
    ):
        self.num_steps = 100
        self.run_id = run_id
        self.seed = random_seed
        self.gm = geometry_manager
        self.center_pos = self.gm.get_solid_center(name="source")
        self.num_particles = num_particles
        self.sim = Simulation(
            self.gm.global_geometry, seed=random_seed, geant4_processes=0
        )
        self.pg = primary_generator(
            self.num_particles, run_id=self.run_id, center_pos=self.center_pos
        )
        self.propagate_photon()

        # 	def __init__(self, geometry_manager, experiment_name, selected_plot, photons, photon_tracks = 1000, run_id = 0, seed = 0, histories = None):

        self.ana_man = analysis_manager(
            self.gm,
            experiment_name,
            plots,
            self.photons,
            self.photon_tracks,
            self.run_id,
            self.seed,
            self.particle_histories,
            write,
        )

    def propagate_photon(self):
        """
        Propagates photons through the geometry using the GPU, collecting their positions and interaction histories.

        :return: None
        """
        nthreads_per_block = 64
        max_blocks = 1024
        seed = 20000000

        gpu_photons = gpu.GPUPhotons(self.pg.primary_photons)
        gpu_geometry = gpu.GPUGeometry(self.gm.global_geometry)
        self.photon_tracks = np.zeros((self.num_steps + 1, self.num_particles, 3))
        self.photon_tracks[0, :, :] = self.pg.positions
        rng_states = gpu.get_rng_states(nthreads_per_block * max_blocks, seed=seed)
        self.interactions = {
            "RAYLEIGH_SCATTER": 4,
            "REFLECT_DIFFUSE": 5,
            "REFLECT_SPECULAR": 6,
            "SURFACE_REEMIT": 7,
            "SURFACE_TRANSMIT": 8,
            "BULK_REEMIT": 9,
            "CHERENKOV": 10,
            "SCINTILLATION": 11,
        }
        self.particle_histories = {
            curr_int: np.zeros(self.num_particles, dtype=int)
            for curr_int in self.interactions.keys()
        }

        for i in range(self.num_steps):
            # propagate photons
            gpu_photons.propagate(
                gpu_geometry,
                rng_states,
                nthreads_per_block=nthreads_per_block,
                max_blocks=max_blocks,
                max_steps=1,
            )
            # from GPUPhoton get event.Photon object
            self.photons = gpu_photons.get()
            self.photon_tracks[i + 1, :, :] = self.photons.pos
            # collect information from flags about interaction history
            self.update_tallies(self.photons)
            # reset interaction history bits that are nonterminal
            new_flags = self.reset_nonterminal_flags(self.photons.flags)
            gpu_photons.flags[: self.num_particles].set(new_flags.astype(np.uint32))

    def reset_nonterminal_flags(self, flag_list):
        """
        Resets non-terminal interaction flags in the given flag list.

        :param flag_list: The list of photon interaction flags.
        :type flag_list: numpy.ndarray
        :return: The updated flag list with non-terminal flags reset.
        :rtype: numpy.ndarray
        """
        # mask = 1111111111111111111000000001111
        mask = 2147479567
        return flag_list & mask

    def update_tallies(self, photons):
        """
        Updates the tally of interactions for the given photons.

        :param photons: The photons whose interaction histories are to be updated.
        :type photons: Photons
        :return: None
        """
        for key, value in self.interactions.items():
            curr_tally = (photons.flags & (0x1 << value)).astype(bool).astype(int)
            self.particle_histories[key] += curr_tally
            # print(key, self.particle_histories[key])


class primary_generator:  # photon generator
    """
    Generates primary photons for the simulation, including their positions, directions, polarization, and wavelengths.

    :param num_particles: The number of particles to generate.
    :type num_particles: int
    :param run_id: The unique identifier for the run.
    :type run_id: int
    :param center_pos: The center position of the photon source.
    :type center_pos: list
    """

    # C++: methods/functions
    # def __init__(self, num_particles, center_pos = [0, 0, 0], delta_placement = 0.0):
    def __init__(self, num_particles, run_id, center_pos=[0, 0, 0]):
        self.num_particles = num_particles
        self.center_pos = center_pos

        # positions

        self.positions = self.get_point_source_pos(
            x0=self.center_pos[0], y0=self.center_pos[1], z0=self.center_pos[2]
        )

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
        self.primary_photons = Photons(
            self.positions, self.directions, self.polarization, self.wavelengths
        )

    def get_point_source_pos(self, x0, y0, z0):
        """
        Generates positions for a point source at the given coordinates.

        :param x0: The x-coordinate of the source.
        :type x0: float
        :param y0: The y-coordinate of the source.
        :type y0: float
        :param z0: The z-coordinate of the source.
        :type z0: float
        :return: The array of positions.
        :rtype: numpy.ndarray
        """
        return np.tile([x0, y0, z0], (self.num_particles, 1))

    # input center of disk and radius
    def get_xy_disk_source_pos(self, x0, y0, z0, r):
        """
        Generates positions for a disk source in the xy-plane with the given center and radius.

        :param x0: The x-coordinate of the disk center.
        :type x0: float
        :param y0: The y-coordinate of the disk center.
        :type y0: float
        :param z0: The z-coordinate of the disk center.
        :type z0: float
        :param r: The radius of the disk.
        :type r: float
        :return: The array of positions.
        :rtype: numpy.ndarray
        """
        # pos_array = np.empty(self.num_particles, 3)
        curr_sqrtr = np.sqrt(np.random.uniform(0, r, self.num_particles))
        curr_theta = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        curr_x = curr_sqrtr * np.cos(curr_theta) + x0
        curr_y = curr_sqrtr * np.sin(curr_theta) + y0
        curr_z = np.ones(self.num_particles) * z0
        return np.vstack((curr_x, curr_y, curr_z)).T

    def get_xz_disk_source_pos(self, x0, y0, z0, r):
        """
        Generates positions for a disk source in the xz-plane with the given center and radius.

        :param x0: The x-coordinate of the disk center.
        :type x0: float
        :param y0: The y-coordinate of the disk center.
        :type y0: float
        :param z0: The z-coordinate of the disk center.
        :type z0: float
        :param r: The radius of the disk.
        :type r: float
        :return: The array of positions.
        :rtype: numpy.ndarray
        """
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
        """
        Generates isotropic directions for the photons.

        :return: The array of directions.
        :rtype: numpy.ndarray
        """
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
        """
        Generates beam directions at the specified angle.

        :param angle: The angle of the beam direction.
        :type angle: float
        :return: The array of beam directions.
        :rtype: numpy.ndarray
        """
        curr_py = np.cos(angle)
        curr_px = np.sin(angle)
        # # curr_pz = 0 shifting the beam in xy plane
        curr_pz = 0
        # curr_pz = np.sin(angle)*np.cos(phi_angle)
        # curr_px = np.sin(angle)*np.sin(phi_angle)
        return np.tile([curr_px, curr_py, curr_pz], (self.num_particles, 1))

    def get_x_cone_dir(self, angle, positive=True):
        """
        Generates directions for a cone in the x direction.

        :param angle: The angle of the cone.
        :type angle: float
        :param positive: Whether the cone is in the positive x direction, defaults to True.
        :type positive: bool, optional
        :return: The array of cone directions.
        :rtype: numpy.ndarray
        """
        phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        cos_theta = np.random.uniform(np.cos(angle / 2), 1, self.num_particles)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        curr_px = cos_theta
        curr_py = np.cos(phi) * sin_theta
        curr_pz = np.sin(phi) * sin_theta
        if not positive:
            curr_px *= -1
        return np.vstack((curr_px, curr_py, curr_pz)).T

    def get_y_cone_dir(self, angle, positive=True):
        """
        Generates directions for a cone in the y direction.

        :param angle: The angle of the cone.
        :type angle: float
        :param positive: Whether the cone is in the positive y direction, defaults to True.
        :type positive: bool, optional
        :return: The array of cone directions.
        :rtype: numpy.ndarray
        """
        phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        cos_theta = np.random.uniform(np.cos(angle / 2), 1, self.num_particles)
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
