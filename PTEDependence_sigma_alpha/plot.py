import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df_spec = pd.read_csv("./results1.csv", names=["specular", "pte", "err"])
df_sig_alpha = pd.read_csv("./results.csv", names=["sigma_alpha", "pte", "err"])

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(15, 10))

# Plot df_spec with x-axis ranging from 0 to 1
ax1.errorbar(1 - df_spec["specular"], df_spec["pte"], yerr=df_spec["err"], label='Specular/Diffuse Reflection Model', fmt='o', color='blue')
ax1.set_xlim(0, 1)  # Set the x-axis limits for df_spec
ax1.set_xlabel('1 - Specular Ratio')
ax1.set_ylabel('PTE')
ax1.set_title('Specular Ratio vs Sigma Alpha Alignment')

# Fit and plot the best-fit line for df_spec
coeff_spec = np.polyfit(1 - df_spec["specular"], df_spec["pte"], 1)  # Linear fit
poly_spec = np.poly1d(coeff_spec)
ax1.plot(1 - df_spec["specular"], poly_spec(1 - df_spec["specular"]), color='blue', linestyle='--', label='Best Fit Specular')

# Calculate R^2 for df_spec
residuals_spec = df_spec["pte"] - poly_spec(1 - df_spec["specular"])
r_squared_spec = 1 - (np.sum(residuals_spec**2) / np.sum((df_spec["pte"] - np.mean(df_spec["pte"]))**2))

# Create a second x-axis for df_sig_alpha
ax2 = ax1.twiny()  # Create a new axis sharing the same y-axis
ax2.errorbar(df_sig_alpha["sigma_alpha"], df_sig_alpha["pte"], yerr=df_sig_alpha["err"], label='Lobed Reflection Model', fmt='o', color='red')
ax2.set_xlim(0, 0.6)  # Set the x-axis limits for df_sig_alpha
ax2.set_xlabel('Sigma Alpha')

# Connect the sigma_alpha points with lines
ax2.plot(df_sig_alpha["sigma_alpha"], df_sig_alpha["pte"], color='red', linestyle='-', marker='o')

# Fit and plot the best-fit line for df_sig_alpha
coeff_alpha = np.polyfit(df_sig_alpha["sigma_alpha"], df_sig_alpha["pte"], 1)  # Linear fit
poly_alpha = np.poly1d(coeff_alpha)

# Calculate R^2 for df_sig_alpha
residuals_alpha = df_sig_alpha["pte"] - poly_alpha(df_sig_alpha["sigma_alpha"])
r_squared_alpha = 1 - (np.sum(residuals_alpha**2) / np.sum((df_sig_alpha["pte"] - np.mean(df_sig_alpha["pte"]))**2))

# Place both legends in the upper right
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Add R^2 values to the plot
ax1.text(0.8, 0.25, f'$R^2$ (Specular): {r_squared_spec:.2f}', transform=ax1.transAxes, fontsize=12, color='blue')

# Save the figure
plt.savefig("./out.png")
plt.show()
