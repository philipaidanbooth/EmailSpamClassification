import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


pp_matrix = np.array([[0.8, 0.2], 
                      [1, 0]])
p_cov = pp_matrix @ pp_matrix.T
e_vals, e_vecs = np.linalg.eig(p_cov)

print(pp_matrix)
print(f"Eigenvectors are:", e_vecs)
print(f"Eigenvalues are:", e_vals)

initial_population = ([100,100])
time = 50
pop_over_time = [initial_population]

for _ in range(50):
    new_pop = pp_matrix @ pop_over_time[-1]
    pop_over_time.append(new_pop)
    
pop_over_time = np.array(pop_over_time)

plt.figure(figsize=(10, 6))
plt.plot(pop_over_time[:, 0], label="Owls (Predators)", color='red', marker='o')
plt.plot(pop_over_time[:, 1], label="Mice (Prey)", color='blue', marker='o')
plt.title("Predator-Prey Model")
plt.xlabel("Time Step")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()
