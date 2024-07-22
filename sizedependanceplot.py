import csv
import matplotlib.pyplot as plt

# Read data from CSV file
r_values = []
pte_values = []
pte_err_values = []

csv_file = "simulation_results.csv"
with open(csv_file, mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        r_values.append(float(row[0]))
        pte_values.append(float(row[1]))
        pte_err_values.append(float(row[2]))

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.errorbar(
    r_values,
    pte_values,
    yerr=pte_err_values,
    fmt="o",
    ecolor="red",
    capsize=5,
    capthick=2,
)
plt.xlabel("Source Radius (r0)")
plt.ylabel("Photon Transmission Efficiency (PTE)")
plt.title("Photon Transmission Efficiency vs Source Radius")
plt.grid(True)
plt.show()
