import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the CSV file
df = pd.read_csv('2d_results.csv')

# Extract the data
n_xe = df['n_xe'].unique()
n_si = df['n_si'].unique()
pte = df['pte'].values.reshape(len(n_xe), len(n_si))

# Create a meshgrid
X, Y = np.meshgrid(n_si, n_xe)

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, pte, cmap='viridis')

# Set labels and title
ax.set_xlabel('n_si')
ax.set_ylabel('n_xe')
ax.set_zlabel('PTE')
ax.set_title('Photon Transmission Efficiency vs Refractive Indices')

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Show the plot
# plt.savfig("2d.png")
plt.show()
