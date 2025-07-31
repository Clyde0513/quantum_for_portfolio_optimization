import pennylane.numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from collections import Counter

# Where I am getting all the information from: https://www.thewiser.org/quantum-portfolio-optimization

# 1.) Create a toy problem data based on https://www.thewiser.org/quantum-portfolio-optimization
n = 6 # Number of bonds (We've picked 6 bonds because it is a small number to work with and visualize easily, as well as to keep the problem simple for demonstration purposes)
np.random.seed(42)  # For reproducibility

# Market Parameters
m = np.random.uniform(0.5, 1.0, size=n)  # M_c (minimum trade)
M = np.random.uniform(1.0, 2.0, size=n)  # M (maximum trade)
i_c = np.random.uniform(0.2, 0.8, size=n)  # i_c (basket inventory)
delta_c = np.random.uniform(0.1, 0.5, size=n)  # delta_c (minimum increment)

# Global parameters
N = 3 # max bonds in a portfolio
rc_min = -0.1  # minimum residual cash flow
rc_max = 0.2  # maximum residual cash flow
mvb = 1.0 # Denominator for cash flow

# One Risk bucket ℓ=0 with one characteristic j=0

beta = np.random.uniform(0.0, 1.0, size=(n,1)) # β_{c,0}
k_target = np.array([[1.0]])  # k_target (target portfolio)
rho = np.array([[5.0]])  # penaty weight on the objective function

lambda_size = 10.0 # λ (penalty parameter)
lambda_RCup = 10.0 # λ_{RC+} (penalty parameter for positive residual cash flow)
lambda_RClo = 10.0 # λ_{RC-} (penalty parameter for negative residual cash flow)
lambda_char = 10.0 # λ_{char} (penalty parameter for characteristic)

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

b_up = 1.5 # Upper bound for characteristic
b_lo = 0.2 # Lower bound for characteristic

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
layers = 4 # Updated from 2 to 4

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