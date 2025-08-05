import pennylane.numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from collections import Counter

# Where I am getting all the information from: https://www.thewiser.org/quantum-portfolio-optimization

# 1.) Create a toy problem data based on https://www.thewiser.org/quantum-portfolio-optimization
n = 6 # umber of bonds (We've picked 6 bonds because it is a small number to work with and visualize easily, as well as to keep the problem simple for demonstration purposes)
np.random.seed(42)  # For reproducibility

# Market Parameters
m = np.random.uniform(0.5, 1.0, size=n)  # M_c (minimum trade)
M = np.random.uniform(1.0, 2.0, size=n)  # M (maximum trade)
i_c = np.random.uniform(0.2, 0.8, size=n)  # i_c (basket inventory)
delta_c = np.random.uniform(0.1, 0.5, size=n)  # delta_c (minimum increment)

# Global parameters
N = 2 # max bonds in a portfolio
rc_min = 0.01  # Reduced from 0.03 to 0.01
rc_max = 0.05  # Increased from 0.08 to 0.05
mvb = 1.0 # Denominator for cash flow

# One Risk bucket ℓ=0 with one characteristic j=0

beta = np.random.uniform(0.0, 1.0, size=(n,1)) # β_{c,0}
k_target = np.array([[1.0]])  # k_target (target portfolio)
rho = np.array([[5.0]])  # penaty weight on the objective function

lambda_size = 30.0 # λ (penalty parameter)
lambda_RCup = 50.0  # Increased from 35.0 to 50.0
lambda_RClo = 50.0  # Increased from 35.0 to 50.0
lambda_char = 25.0 # λ_{char} (penalty parameter for characteristic)

# Pre-compute x_c based on the given formula: # x_c = (m + min(M, i_c)) / 2 * delta_c
# Here, x_c (how much of bond c is included in the basket) is not a variable, but is fixed to the average value it is allowed to have if c is included at all in the portfolio
x_c = (m + np.minimum(M, i_c)) / 2 * delta_c
# - This is a fixed value based on the market parameters and does not change during optimization.
# - delta_c is the minimum increment, so we multiply by it to ensure that the amount included in the basket is a multiple of this increment.

# 2.) Build Q, q from penalties
Q = np.zeros((n, n))  # Initialize Q matrix
q = np.zeros(n)  # Initialize q vector

# Main Objective: min Σ_l in L  Σ_j in J (ρ_j * (Σ β_c,j * x_c - K_target_l,j)^2

l, j = 0, 0
w = rho[l]
k_target_lj = k_target[l, j]
# Extract scalar values explicitly
for a in range(n):
    for b in range(n):
        Q[a, b] += w.item() * beta[a, j].item() * beta[b, j].item() * x_c[a].item() * x_c[b].item()
for a in range(n):
    q[a] += -2 * w.item() * beta[a, j].item() * x_c[a].item() * k_target_lj.item()

# Basket-size constraint: (Σ y - N)²

Q += lambda_size * np.ones((n, n))  # Add penalty for basket size
q += -2 * lambda_size * N * np.ones(n)  # Adjust q for basket size constraint

# Residual cash flow <= rc_max and >= rc_min

a_cf = ((m * delta_c) / (100 * mvb)) * x_c  # a_cf (cash flow coefficient)

# Upper-bound constraint: Σ y * a_cf <= rc_max
Q += lambda_RCup * np.outer(a_cf, a_cf)  # Add penalty for upper bound
q += -2 * lambda_RCup * rc_max * a_cf  # Adjust q for upper bound

# Lower-bound constraint: Σ y * a_cf >= rc_min
Q += lambda_RClo * np.outer(a_cf, a_cf)  # Add penalty for lower bound
q += -2 * lambda_RClo * rc_min * a_cf  # Adjust q for lower bound

# Characteristic bound (same bucket as l = 0, j = 0)
# (Σ β k y ≤ b_up) and (≥ b_lo)

b_up = 1.0 # Upper bound for characteristic
b_lo = 0.6 # Lower bound for characteristic

# <= b_up
Q += lambda_char * np.outer(beta[:, j]*i_c, beta[:, j]*i_c)  # Add penalty for upper bound
q += -2 * lambda_char * b_up * (beta[:, j] * i_c)  # Adjust q for upper bound

# >= b_lo
Q += lambda_char * np.outer(beta[:, j]*i_c, beta[:, j]*i_c)  # Add penalty for lower bound
q += -2 * lambda_char * b_lo * (beta[:, j] * i_c)  # Adjust q for lower bound

# Symmetrize Q matrix
Q = (Q + Q.T) / 2

# Map QUBO problem to Pennylane Hamiltonian

coefficients, obs = [],[]

for i in range(n):
    # Linear terms
    coefficients += [q[i]/2, -q[i]/2]
    obs += [qml.Identity(i), qml.PauliZ(i)]
    
    # Quadratic terms
    for j in range(i+1, n):
        Q_if = Q[i, j]
        coefficients += [Q_if/4, -Q_if/4, -Q_if/4, Q_if/4]
        obs += [qml.Identity(i), qml.PauliZ(i), qml.PauliZ(j), qml.PauliZ(i) @ qml.PauliZ(j)]
        
Hamiltonian_cost = qml.Hamiltonian(coefficients, obs)


# 3.) VQE with StrongEntanglinglayers
dev = qml.device("default.qubit", wires=n)
layers = 6 # Updated from 4 to 6

def ansatz(params, wires):
    qml.StronglyEntanglingLayers(params, wires=wires)

# This is the Cost function for the VQE
# - <qml.PauliZ> observables
# - The circuit is defined using the ansatz function
# - The circuit is executed using qml.qnode decorator
@qml.qnode(dev)
def circuit(params):
    ansatz(params, wires=range(n))
    return qml.expval(Hamiltonian_cost)

# Initialize parameters
params = np.random.uniform(0, 2 * np.pi, (layers, n, 3), requires_grad=True)

optimizer = qml.AdamOptimizer(stepsize=0.1)
for it in range(100):
    params, cost = optimizer.step_and_cost(circuit, params)
    if it % 10 == 0:
        print(f"Iteration {it}, Cost: {cost}")
        
        
# 4.) Sample and Interpret Results

dev = qml.device("default.qubit", wires=n, shots=5000)  # Updated from 1000 to 5000

# Fix ValueError by applying qml.PauliZ to each wire individually
@qml.qnode(dev)
def sample_circuit(params):
    ansatz(params, wires=range(n))
    return [qml.sample(qml.PauliZ(w)) for w in range(n)]

# Convert samples to a NumPy array for arithmetic operations
samples = np.array(sample_circuit(params))
bit_strings = ((1 - samples) // 2).astype(int)  # Convert to 0s and 1s
counts = Counter(map(tuple, bit_strings.tolist())) # Count occurrences of each bit string
# Sort counts by frequency

best, freq = counts.most_common(1)[0]  # Get the most common bit string and its frequency
print(f"Best portfolio (bit string (y) ): {best}, Frequency: {freq}")

# Convert counts to a sorted list of bit strings and their frequencies
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
bit_strings, frequencies = zip(*sorted_counts)

# Normalize frequencies to percentages
total_samples = sum(frequencies)
percentages = [freq / total_samples * 100 for freq in frequencies]

# Convert bit strings to strings for better visualization
bit_strings = [''.join(map(str, bs)) for bs in bit_strings]

# Truncate long bit strings for better readability
truncated_bit_strings = [f"{bs[:3]}...{bs[-3:]}" if len(bs) > 6 else bs for bs in bit_strings]

# Plot bar chart with truncated bit strings
plt.figure(figsize=(12, 6))
plt.bar(truncated_bit_strings, percentages, color='blue', alpha=0.7, label='Portfolios')

# Highlight the best portfolio
best_index = bit_strings.index(''.join(map(str, best)))
plt.bar(truncated_bit_strings[best_index], percentages[best_index], color='red', label='Best Portfolio')

# Add labels and title
plt.xlabel('Bit Strings (Portfolios)')
plt.ylabel('Frequency (%)')
plt.title('Portfolio Frequencies (Normalized)')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Add cumulative frequency plot with truncated bit strings
cumulative_percentages = np.cumsum(percentages)
plt.figure(figsize=(12, 6))
plt.plot(truncated_bit_strings, cumulative_percentages, marker='o', color='green', label='Cumulative Frequency')

# Add labels and title
plt.xlabel('Bit Strings (Portfolios)')
plt.ylabel('Cumulative Frequency (%)')
plt.title('Cumulative Portfolio Frequencies')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Display top portfolios in a table-like format with truncated bit strings
print("Top Portfolios:")
print(f"{'Portfolio':<15}{'Frequency':<15}{'Percentage (%)':<15}")
for i in range(min(10, len(truncated_bit_strings))):
    print(f"{truncated_bit_strings[i]:<15}{frequencies[i]:<15}{percentages[i]:<15.2f}")

# Extract unique portfolio configurations
unique_bit_strings = np.unique(samples.T, axis=0)  # Transpose samples to align dimensions

# Calculate residual cash flow and basket size for each unique portfolio
residual_cash_flows = [np.dot(a_cf, bs) for bs in unique_bit_strings]
basket_sizes = [np.sum(bs) for bs in unique_bit_strings]

# Generate labels for unique bit strings
unique_truncated_bit_strings = [''.join(map(str, bs)) for bs in unique_bit_strings]

# Plot residual cash flow distribution
plt.figure(figsize=(12, 6))
plt.bar(unique_truncated_bit_strings, residual_cash_flows, color='purple', alpha=0.7, label='Residual Cash Flow')
plt.axhline(rc_min, color='red', linestyle='--', label='rc_min')
plt.axhline(rc_max, color='green', linestyle='--', label='rc_max')
plt.xlabel('Bit Strings (Portfolios)')
plt.ylabel('Residual Cash Flow')
plt.title('Residual Cash Flow Distribution')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Plot basket size distribution
plt.figure(figsize=(12, 6))
plt.bar(unique_truncated_bit_strings, basket_sizes, color='orange', alpha=0.7, label='Basket Size')
plt.axhline(N, color='blue', linestyle='--', label='Max Basket Size (N)')
plt.xlabel('Bit Strings (Portfolios)')
plt.ylabel('Basket Size')
plt.title('Basket Size Distribution')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()