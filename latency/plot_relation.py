import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 100
# l = [1, 10]
# c = [1]
l = []
c = []
bandwidth_kbs = 1e6
message_size = 1
latency =  1e-5
l.append(latency)
c.append(message_size / bandwidth_kbs)
# Define the range of k
k_values = np.arange(1, 21)  # k ranges from 1 to 20

# Define the function to be plotted
def f(k, N, l, c):
    latency = l + (k) * c
    return np.log(N) / np.log(k) * latency

# Plotting
plt.figure(figsize=(10, 10))
for i in range(len(l)):
    for j in range(len(c)):
        # Compute function values
        function_values = f(k_values, N, l[i], c[j])        
        plt.plot(k_values, function_values, marker='o', label=f'l={l[i]}, c={c[j]}')
plt.title('Plot of the function $f(k) = \\log_k(N)(l + k \\cdot c)$')
plt.xlabel('$k$')
plt.ylabel('$f(k)$')
plt.grid(True)
plt.legend()
plt.savefig('relation.png')