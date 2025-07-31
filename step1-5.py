import pennylane.numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from collections import Counter
import scipy.optimize as opt

# Reference: https://www.thewiser.org/quantum-portfolio-optimization

print("=== Quantum Portfolio Optimization ===\n")

# 1.) Create toy problem data based on mathematical formulation
n = 6  # number of bonds
np.random.seed(42)  # For reproducibility

# Market Parameters
m = np.random.uniform(0.5, 1.0, size=n)  # m_c (minimum trade)
M = np.random.uniform(1.0, 2.0, size=n)  # M_c (maximum trade)
i_c = np.random.uniform(0.2, 0.8, size=n)  # i_c (basket inventory)
delta_c = np.random.uniform(0.1, 0.5, size=n)  # delta_c (minimum increment)

# Global parameters
N = 3  # max bonds in portfolio
rc_min = 0.01  # Minimum residual cash flow
rc_max = 0.05  # Maximum residual cash flow
mvb = 1.0  # Market value base

# Risk characteristics (one bucket ℓ=0, one characteristic j=0)
beta = np.random.uniform(0.0, 1.0, size=(n, 1))  # β_{c,0}
k_target = np.array([[1.0]])  # Target for characteristic
rho = np.array([[5.0]])  # Penalty weight

# Enhanced penalty parameters (much more aggressive for constraint satisfaction)
lambda_size = 2000.0   # Basket size constraint penalty 
lambda_RCup = 500.0    # Upper cash flow constraint penalty  
lambda_RClo = 500.0    # Lower cash flow constraint penalty
lambda_char = 300.0    # Characteristic constraint penalty 

# Pre-compute x_c (fixed amount of bond c if included)
x_c = (m + np.minimum(M, i_c)) / (2 * delta_c)  # Fixed value based on market parameters
print(f"Bond amounts if selected (x_c): {x_c}")
print(f"Target basket size: {N}")
print(f"Cash flow range: [{rc_min:.4f}, {rc_max:.4f}]")
print(f"Characteristic range: [{0.6:.4f}, {1.0:.4f}]")

# 2.) Build QUBO formulation: min y^T Q y + q^T y
Q = np.zeros((n, n))
q = np.zeros(n)

print("\nBuilding QUBO formulation...")

# Main Objective: Minimize tracking error
# min ρ * (Σ β_{c,j} * x_c * y_c - K_target)^2
l, j = 0, 0
w = rho[l, j]
k_target_lj = k_target[l, j]

# Quadratic terms: w * β_i * β_j * x_i * x_j * y_i * y_j
for i in range(n):
    for k in range(n):
        Q[i, k] += w * beta[i, j] * beta[k, j] * x_c[i] * x_c[k]

# Linear terms: -2 * w * k_target * β_i * x_i * y_i
for i in range(n):
    q[i] += -2 * w * beta[i, j] * x_c[i] * k_target_lj

# Basket size constraint: (Σ y_c - N)^2
# Expands to: Σ y_c^2 + Σ_{i≠j} y_i*y_j - 2*N*Σ y_c + N^2
Q += lambda_size * np.ones((n, n))  # Cross terms
np.fill_diagonal(Q, np.diag(Q) - lambda_size)  # Remove diagonal double-counting
q += -2 * lambda_size * N * np.ones(n)

# Cash flow coefficients
a_cf = (m * delta_c * x_c) / (100 * mvb)
print(f"Cash flow coefficients (a_cf): {a_cf}")

# Upper cash flow constraint: (Σ a_cf * y_c - rc_max)^2 if Σ a_cf * y_c > rc_max
Q += lambda_RCup * np.outer(a_cf, a_cf)
q += -2 * lambda_RCup * rc_max * a_cf

# Lower cash flow constraint: (rc_min - Σ a_cf * y_c)^2 if Σ a_cf * y_c < rc_min
Q += lambda_RClo * np.outer(a_cf, a_cf)
q += -2 * lambda_RClo * rc_min * a_cf

# Characteristic bounds
b_up = 1.0  # Upper bound
b_lo = 0.6  # Lower bound
char_coeff = beta[:, j] * i_c

# Upper characteristic constraint: (Σ β*i_c*y - b_up)^2 if > b_up
Q += lambda_char * np.outer(char_coeff, char_coeff)
q += -2 * lambda_char * b_up * char_coeff

# Lower characteristic constraint: (b_lo - Σ β*i_c*y)^2 if < b_lo  
Q += lambda_char * np.outer(char_coeff, char_coeff)
q += -2 * lambda_char * b_lo * char_coeff

# Symmetrize Q matrix
Q = (Q + Q.T) / 2

print(f"QUBO matrix Q shape: {Q.shape}")
print(f"QUBO vector q shape: {q.shape}")

# 3.) Convert QUBO to Hamiltonian (mapping)
coefficients = []
obs = []
constant = 0.0

# Linear terms: q_i * x_i = q_i * (1 - z_i)/2 = q_i/2 - q_i*z_i/2
for i in range(n):
    constant += q[i] / 2
    coefficients.append(-q[i] / 2)
    obs.append(qml.PauliZ(i))

# Quadratic terms: Q_ij * x_i * x_j = Q_ij * (1-z_i)/2 * (1-z_j)/2
for i in range(n):
    for j in range(i, n):  # Include diagonal terms
        coeff = Q[i, j]
        if i == j:
            # Diagonal terms: Q_ii * x_i^2 = Q_ii * x_i (since x_i^2 = x_i for binary)
            constant += coeff / 4
            coefficients.append(-coeff / 4)
            obs.append(qml.PauliZ(i))
        else:
            # Off-diagonal terms
            constant += coeff / 4
            coefficients.append(-coeff / 4)
            obs.append(qml.PauliZ(i))
            coefficients.append(-coeff / 4)
            obs.append(qml.PauliZ(j))
            coefficients.append(coeff / 4)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

# Add constant term
obs.append(qml.Identity(0))
coefficients.append(constant)

Hamiltonian_cost = qml.Hamiltonian(coefficients, obs)
print(f"\nHamiltonian constructed with {len(coefficients)} terms")

# Define classical objective function (needed for analysis)
def classical_objective(x):
    """Evaluate the original QUBO objective"""
    return x.T @ Q @ x + q.T @ x

# 4.) VQE Implementation with Multiple Restarts
dev = qml.device("default.qubit", wires=n)
layers = 8  # Increased number of layers in the ansatz

def ansatz(params, wires):
    # Initialize with equal superposition
    for wire in wires:
        qml.Hadamard(wire)
    # Apply parameterized layers
    qml.StronglyEntanglingLayers(params, wires=wires)

@qml.qnode(dev)
def circuit(params):
    ansatz(params, wires=range(n))
    return qml.expval(Hamiltonian_cost)

print("\nStarting VQE optimization with multiple restarts...")

best_params = None
best_cost = float('inf')
all_costs = []

# Multiple random restarts
num_restarts = 3
for restart in range(num_restarts):
    print(f"\n--- Restart {restart + 1}/{num_restarts} ---")
    
    # Initialize parameters with different random seeds
    np.random.seed(42 + restart * 10)
    params = np.random.uniform(0, 2 * np.pi, (layers, n, 3), requires_grad=True)
    
    optimizer = qml.AdamOptimizer(stepsize=0.03)  # Slightly reduced step size
    costs = []
    
    for it in range(250):  # More iterations per restart
        params, cost = optimizer.step_and_cost(circuit, params)
        costs.append(cost)
        if it % 50 == 0:
            print(f"  Iteration {it:3d}, Cost: {cost:8.4f}")
    
    all_costs.extend(costs)
    final_cost = costs[-1]
    print(f"  Final cost for restart {restart + 1}: {final_cost:.4f}")
    
    if final_cost < best_cost:
        best_cost = final_cost
        best_params = params.copy()
        print(f"  *** New best cost: {best_cost:.4f} ***")

print(f"\nBest VQE cost across all restarts: {best_cost:.4f}")
params = best_params

# 5.) Sample and interpret results
print("\nSampling from optimized circuit...")
dev_sample = qml.device("default.qubit", wires=n, shots=20000)  # More samples

@qml.qnode(dev_sample)
def sample_circuit(params):
    ansatz(params, wires=range(n))
    return [qml.sample(qml.PauliZ(w)) for w in range(n)]

# Convert Pauli-Z samples to binary
samples = np.array(sample_circuit(params))
bit_strings = ((1 - samples) // 2).astype(int)
counts = Counter(map(tuple, bit_strings.T))

# Get best solution (convert to regular ints)
best_solution, best_freq = counts.most_common(1)[0]
best_solution = [int(x) for x in best_solution]  # Convert tensors to ints
print(f"\nMost frequent quantum solution: {best_solution}")
print(f"Frequency: {best_freq}/{sum(counts.values())} ({100*best_freq/sum(counts.values()):.1f}%)")

# Also check for best feasible solution among top results
print(f"\nAnalyzing top quantum solutions for constraint satisfaction...")
best_feasible_quantum = None
best_feasible_cost = float('inf')
all_feasible_solutions = []

for solution, freq in counts.most_common(20):  # Check top 20
    sol_array = np.array([int(x) for x in solution])
    if sol_array.sum() == N:  # Exactly N bonds (feasible)
        cost = classical_objective(sol_array)
        all_feasible_solutions.append((sol_array, cost, freq))
        if cost < best_feasible_cost:
            best_feasible_cost = cost
            best_feasible_quantum = sol_array
            print(f"  Found feasible quantum solution: {[int(x) for x in solution]} (cost: {cost:.2f}, freq: {freq})")

# Sort all feasible solutions by cost
all_feasible_solutions.sort(key=lambda x: x[1])

if best_feasible_quantum is not None:
    print(f"\nUsing best feasible quantum solution instead of most frequent:")
    best_solution = best_feasible_quantum
    print(f"Best feasible quantum solution: {[int(x) for x in best_solution]}")
    print(f"Found {len(all_feasible_solutions)} total feasible quantum solutions")
else:
    print(f"\nNo feasible solutions found, using most frequent solution.")

# 6.) Classical benchmark for comparison
print("\nSolving classically for benchmark...")

# Brute force for small problems
best_classical_cost = float('inf')
best_classical_solution = None

for i in range(2**n):
    x = np.array([(i >> j) & 1 for j in range(n)])
    cost = classical_objective(x)
    if cost < best_classical_cost:
        best_classical_cost = cost
        best_classical_solution = x

print(f"Best classical solution: {tuple(best_classical_solution)}")
print(f"Classical cost: {best_classical_cost:.4f}")

# Evaluate quantum solution
quantum_cost = classical_objective(np.array(best_solution))
print(f"Quantum solution cost: {quantum_cost:.4f}")

# 7.) Constraint satisfaction analysis
def analyze_solution(solution, name):
    sol = np.array(solution)
    basket_size = sol.sum()
    cash_flow = np.dot(a_cf, sol)
    characteristic = np.dot(char_coeff, sol)
    tracking_error = abs(np.dot(beta[:, 0] * x_c, sol) - k_target_lj)
    
    # Check constraint violations
    basket_violation = abs(basket_size - N)
    cash_violation = max(0, rc_min - cash_flow) + max(0, cash_flow - rc_max)
    char_violation = max(0, b_lo - characteristic) + max(0, characteristic - b_up)
    
    print(f"\n{name} Solution Analysis:")
    print(f"  Portfolio: {[int(x) for x in solution]}")
    print(f"  Basket size: {basket_size} (target: {N}) - Violation: {basket_violation}")
    print(f"  Cash flow: {cash_flow:.4f} (range: [{rc_min:.4f}, {rc_max:.4f}]) - Violation: {cash_violation:.4f}")
    print(f"  Characteristic: {characteristic:.4f} (range: [{b_lo:.4f}, {b_up:.4f}]) - Violation: {char_violation:.4f}")
    print(f"  Tracking error: {tracking_error:.4f}")
    
    total_violation = basket_violation + cash_violation + char_violation
    print(f"  Total constraint violation: {total_violation:.4f}")
    
    return total_violation

print(f"\n=== Detailed Solution Analysis ===")
classical_violation = analyze_solution(best_classical_solution, "Classical")
quantum_violation = analyze_solution(best_solution, "Quantum")

# Compare all solutions with exactly N bonds
print(f"\n=== Analyzing Feasible Solutions (exactly {N} bonds) ===")
feasible_solutions = []
for i in range(2**n):
    x = np.array([(i >> j) & 1 for j in range(n)])
    if x.sum() == N:
        cost = classical_objective(x)
        feasible_solutions.append((x, cost))

if feasible_solutions:
    feasible_solutions.sort(key=lambda x: x[1])
    best_feasible = feasible_solutions[0]
    print(f"Best feasible solution: {[int(x) for x in best_feasible[0]]}")
    print(f"Best feasible cost: {best_feasible[1]:.4f}")
    analyze_solution(best_feasible[0], "Best Feasible")
else:
    print("No feasible solutions with exactly N bonds found!")

# Summary
print(f"\n=== Final Summary ===")
print(f"Problem size: {n} bonds, Target basket: {N}")
print(f"VQE layers: {layers}, Restarts: {num_restarts}, Iterations per restart: 250")
print(f"Best quantum cost: {quantum_cost:.4f} (violation: {quantum_violation:.4f})")
print(f"Best classical cost: {best_classical_cost:.4f} (violation: {classical_violation:.4f})")
print(f"Cost ratio (quantum/classical): {quantum_cost/best_classical_cost:.3f}")

# Determine the winner
if quantum_violation < classical_violation:
    print("QUANTUM WINS: Better constraint satisfaction")
    advantage = "constraint satisfaction"
elif quantum_cost < best_classical_cost:
    print("QUANTUM WINS: Better objective value")
    advantage = "objective value" 
elif abs(quantum_cost - best_classical_cost) < 1e-6 and quantum_violation <= classical_violation:
    print("QUANTUM WINS: Tied objective with better/equal constraints")
    advantage = "tie-breaking"
else:
    print("Classical solution dominates quantum solution")
    advantage = None

if advantage:
    print(f"\nQUANTUM ADVANTAGE ACHIEVED through {advantage}")
    print(f"   Quantum: {[int(x) for x in best_solution]} (violations: {quantum_violation:.4f})")
    print(f"   Classical: {[int(x) for x in best_classical_solution]} (violations: {classical_violation:.4f})")

print(f"\nTop 5 quantum solutions:")
for i, (solution, freq) in enumerate(counts.most_common(5)):
    cost = classical_objective(np.array(solution))
    violation = sum([int(x) for x in solution]) != N  # Simplified violation check
    print(f"  {i+1}. {[int(x) for x in solution]} (freq: {freq:4d}, cost: {cost:7.2f}, feasible: {not violation})")
