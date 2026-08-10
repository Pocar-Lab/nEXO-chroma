from PocarChroma import geometry_manager as gm
from PocarChroma import plotter
from PocarChroma import photons as ph
from PocarChroma import save_load_sim as sls

import pandas as pd
import numpy as np
import os
import h5py
import time
import random
import sys

### Inputs

experiment_name = 'CHANGE_ME'

num_photons = 10_000_000
batch_size = 1_000_000
save_tracks = 1_0000
num_steps = 15
seed = 9050
exclude = [] # List of parts to exclude from geometry

### Configure Geometry

geometry = gm.geometry_manager(
    geometry_data_path='parts.csv',
    surface_data_path='surfaces.csv',
    material_data_path='materials.csv',
    exclude=exclude,
)
global_geometry = geometry.global_geometry
source_location = geometry.get_solid_center('source')

### Setup Filters

detected = ph.Filter(geometry, ph.Interaction.SURFACE_DETECT)

### Run Simulation

start = time.time()
for photons in ph.photon_generator(seed, num_photons, source_location=source_location):
    print(f"Simulating {len(photons.pos):,} photons")
    # Propagate batch of photons
    photon_tracks = ph.propagate(
        seed=seed,
        photons=photons,
        geometry= geometry,
        track_return_ct=0,
        num_steps=num_steps
        )

    # Update filters
    detected.update(photon_tracks)
end = time.time()

print(f"Elapsed time: {end - start:.4} seconds")

### Analyze Results

print(f"PTE: {len(detected.res)/num_photons}")

### Plot results

# Plot geometry and photon tracks
plotter.plot_geometry_tracks(geometry.geometry_df, photon_tracks)
# Plot geometry and photon tracks which are detected
plotter.plot_geometry_tracks(geometry.geometry_df, photon_tracks, [detected.res], 400, ['blue'])
