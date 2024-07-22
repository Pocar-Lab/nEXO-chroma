import csv
import matplotlib.pyplot as plt
import numpy as np
# Read data from CSV file
r_values = []
pte_values = []
pte_err_values = []

csv_file = "simulation_results_8reflector.csv"
with open(csv_file, mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        r_values.append(float(row[0]))
        pte_values.append(float(row[1]))
        pte_err_values.append(float(row[2]))

r_values_noreflector = []
pte_values_noreflector = []
pte_err_values_noreflector = []

csv_file = "simulation_results_noreflector.csv"
with open(csv_file, mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        r_values_noreflector.append(float(row[0]))
        pte_values_noreflector.append(float(row[1]))
        pte_err_values_noreflector.append(float(row[2]))

import matplotlib.pyplot as plt
import numpy as np

# Assuming pte_values, pte_err_values, pte_values_noreflector, pte_err_values_noreflector, and r_values are defined

pte = np.asarray(pte_values)
pte_noreflector = np.asarray(pte_values_noreflector)
pte_ratio = pte / pte_noreflector

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: pte_values with error bars
axs[0, 0].errorbar(
    r_values,
    pte,
    yerr=pte_err_values,
    fmt="o",
    color="blue",
    ecolor="red",
    capsize=5,
    capthick=2,
)
axs[0, 0].set_xlabel("Source Radius (r0)")
axs[0, 0].set_ylabel("Photon Transmission Efficiency (PTE)")
axs[0, 0].set_title("PTE with Reflector")
axs[0, 0].grid(True)

# Plot 2: pte_values_noreflector with error bars
axs[0, 1].errorbar(
    r_values,
    pte_noreflector,
    yerr=pte_err_values_noreflector,
    fmt="o",
    color="green",
    ecolor="red",
    capsize=5,
    capthick=2,
)
axs[0, 1].set_xlabel("Source Radius (r0)")
axs[0, 1].set_ylabel("Photon Transmission Efficiency (PTE)")
axs[0, 1].set_title("PTE without Reflector")
axs[0, 1].grid(True)

# Plot 3: pte_values/pte_values_noreflector with error bars
pte_ratio_err = pte_ratio * np.sqrt((pte_err_values/pte)**2 + (pte_err_values_noreflector/pte_noreflector)**2)
axs[1, 0].errorbar(
    r_values,
    pte_ratio,
    yerr=pte_ratio_err,
    fmt="o",
    color="purple",
    ecolor="red",
    capsize=5,
    capthick=2,
)
axs[1, 0].set_xlabel("Source Radius (r0)")
axs[1, 0].set_ylabel("PTE Ratio (With 8/Without Reflector)")
axs[1, 0].set_title("PTE Ratio with 8/without Reflector")
axs[1, 0].grid(True)

# Hide the 4th subplot
axs[1, 1].axis('off')

fig.suptitle("PTE vs Source Radius", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

