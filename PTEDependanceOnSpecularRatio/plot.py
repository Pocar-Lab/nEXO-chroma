import pandas as pd
import matplotlib.pyplot as plt 



df = pd.read_csv("./results.csv", names = ["specular", "pte", "err"])
print(df)

plt.figure(figsize=(15, 10))
plt.errorbar(df["specular"], df["pte"], df["err"])
plt.savefig("./out.png")