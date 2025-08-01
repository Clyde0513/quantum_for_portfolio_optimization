import pennylane.numpy as np
import pennylane as qml
from collections import Counter
import scipy.optimize as opt
import time

print("=== Exploring (part 1) Quantum Portfolio Optimization ===\n")

# 1. Optimize Problem Size for Memory Constraints
n = 12
np.random.seed(42)

# Market Parameters
m = np.random.uniform(0.5, 1.0, size=n)
M = np.random.uniform(1.0, 2.0, size=n)
i_c = np.random.uniform(0.2, 0.8, size=n)
delta_c = np.random.uniform(0.1, 0.5, size=n)

# Global Parameters
N = 6  # Adjusted to half of n for reasonable basket size
rc_min = 0.01
rc_max = 0.05
mvb = 1.0

# Risk Characteristics
beta = np.random.uniform(0.0, 1.0, size=(n, 1))
k_target = np.array([[1.0]])
rho = np.array([[5.0]])

# Initial Penalty Parameters (will be adjusted adaptively)
lambda_size = 2000.0
lambda_RCup = 500.0
lambda_RClo = 500.0
lambda_char = 300.0

# Pre-compute x_c
x_c = (m + np.minimum(M, i_c)) / (2 * delta_c)
print(f"Bond amounts if selected (x_c): {x_c}")
print(f"Target basket size: {N}")
print(f"Cash flow range: [{rc_min:.4f}, {rc_max:.4f}]")
print(f"Characteristic range: [0.6, 1.2]")

# 2. Build QUBO Formulation
Q = np.zeros((n, n))
q = np.zeros(n)
print("\nBuilding QUBO formulation...")

# Tracking Error
l, j = 0, 0
w = rho[l, j]
k_target_lj = k_target[l, j]
for i in range(n):
    for k in range(n):
        Q[i, k] += w * beta[i, j] * beta[k, j] * x_c[i] * x_c[k]
for i in range(n):
    q[i] += -2 * w * beta[i, j] * x_c[i] * k_target_lj

# Basket Size Constraint
Q += lambda_size * np.ones((n, n))
np.fill_diagonal(Q, np.diag(Q) - lambda_size)
q += -2 * lambda_size * N * np.ones(n)

# Cash Flow Coefficients
a_cf = (m * delta_c * x_c) / (100 * mvb)
print(f"Cash flow coefficients (a_cf): {a_cf}")

# Cash Flow Constraints
Q += lambda_RCup * np.outer(a_cf, a_cf)
q += -2 * lambda_RCup * rc_max * a_cf
Q += lambda_RClo * np.outer(a_cf, a_cf)
q += -2 * lambda_RClo * rc_min * a_cf

# Characteristic Bounds
b_up = 1.2
b_lo = 0.6
char_coeff = beta[:, j] * i_c
Q += lambda_char * np.outer(char_coeff, char_coeff)
q += -2 * lambda_char * b_up * char_coeff
Q += lambda_char * np.outer(char_coeff, char_coeff)
q += -2 * lambda_char * b_lo * char_coeff

# Symmetrize Q
Q = (Q + Q.T) / 2
print(f"QUBO matrix Q shape: {Q.shape}")

# Classical Objective Function
def classical_objective(x):
    return x.T @ Q @ x + q.T @ x

# Constraint Violation Function for Adaptive Penalties
def compute_violations(x):
    """Compute constraint violations for the given solution x."""
    basket_size = x.sum() - N
    cash_flow = np.dot(a_cf, x)
    characteristic = np.dot(char_coeff, x)
    basket_violation = basket_size ** 2
    cash_violation = max(0, rc_min - cash_flow) ** 2 + max(0, cash_flow - rc_max) ** 2
    char_violation = max(0, b_lo - characteristic) ** 2 + max(0, characteristic - b_up) ** 2
    return basket_violation, cash_violation, char_violation

# 3. Convert QUBO to Hamiltonian
coefficients = []
obs = []
constant = 0.0
for i in range(n):
    constant += q[i] / 2
    coefficients.append(-q[i] / 2)
    obs.append(qml.PauliZ(i))
for i in range(n):
    for j in range(i, n):
        coeff = Q[i, j]
        if i == j:
            constant += coeff / 4
            coefficients.append(-coeff / 4)
            obs.append(qml.PauliZ(i))
        else:
            constant += coeff / 4
            coefficients.append(-coeff / 4)
            obs.append(qml.PauliZ(i))
            coefficients.append(-coeff / 4)
            obs.append(qml.PauliZ(j))
            coefficients.append(coeff / 4)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
            
obs.append(qml.Identity(0))
coefficients.append(constant)
Hamiltonian_cost = qml.Hamiltonian(coefficients, obs)
print(f"\nHamiltonian constructed with {len(coefficients)} terms")

# 4. VQE with QAOA Ansatz and Adaptive Penalties
try:
    dev = qml.device("lightning.qubit", wires=n)
    print("Using lightning.qubit device for CPU optimization")
except:
    dev = qml.device("default.qubit", wires=n)
    print("Using default.qubit device")

p = 4 # QAOA layers (reduced from 6 to save memory and computation)

def qaoa_ansatz(params, wires):
    gammas = params[:p]
    betas = params[p:]
    for wire in range(n):
        qml.Hadamard(wire)
    for layer in range(p):
        for i in range(n):
            for j in range(i + 1, n):
                if Q[i, j] != 0:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * gammas[layer] * Q[i, j], wires=j)
                    qml.CNOT(wires=[i, j])
        for i in range(n):
            qml.RZ(2 * gammas[layer] * q[i], wires=i)
        for i in range(n):
            qml.RX(2 * betas[layer], wires=i)

    # # Define the QAOA circuit
    # for wire in range(n):
    #     qml.RZ(2 * params[-1], wires=wire)  # Final rotation for the last parameter
    
        
@qml.qnode(dev)
def circuit(params):
    qaoa_ansatz(params, wires=range(n))
    return qml.expval(Hamiltonian_cost)

# Adaptive Penalty Adjustment
def update_penalties(solution, lambda_size, lambda_RCup, lambda_RClo, lambda_char):
    basket_violation, cash_violation, char_violation = compute_violations(solution)
    if basket_violation > 1e-2:
        lambda_size *= 1.5
    if cash_violation > 1e-2:
        lambda_RCup *= 1.2
        lambda_RClo *= 1.2
    if char_violation > 1e-2:
        lambda_char *= 1.2
    return lambda_size, lambda_RCup, lambda_RClo, lambda_char

print("\nStarting VQE optimization with QAOA and adaptive penalties...")
best_params = None
best_cost = float('inf')
num_restarts = 5  # Increased restarts
for restart in range(num_restarts):
    print(f"\n--- Restart {restart + 1}/{num_restarts} ---")
    np.random.seed(42 + restart * 10)
    params = np.random.uniform(0, np.pi, 2 * p)  # QAOA parameters: gammas and betas
    optimizer = qml.AdamOptimizer(stepsize=0.02)
    costs = []
    current_lambda_size, current_lambda_RCup, current_lambda_RClo, current_lambda_char = (
        lambda_size, lambda_RCup, lambda_RClo, lambda_char
    )
    for it in range(500):  # Increased iterations
        params, cost = optimizer.step_and_cost(circuit, params)
        costs.append(cost)
        if it % 100 == 0:
            print(f" Iteration {it:3d}, Cost: {cost:8.4f}")
        # Adaptive penalties every 50 iterations
        if it % 50 == 0 and it > 0:
            try:
                dev_temp = qml.device("lightning.qubit", wires=n, shots=100)
            except:
                dev_temp = qml.device("default.qubit", wires=n, shots=100)
            
            # Sample from the circuit to get a solution
            # qml.set_device(dev_temp)
            @qml.qnode(dev_temp)
            def temp_circuit():
                qaoa_ansatz(params, wires=range(n))
                return [qml.sample(qml.PauliZ(w)) for w in range(n)]
            
            samples = np.array(temp_circuit())
            bit_strings = ((1 - samples) // 2).astype(int)
            most_common = Counter(map(tuple, bit_strings.T)).most_common(1)[0][0]
            solution = np.array([int(x) for x in most_common])
            current_lambda_size, current_lambda_RCup, current_lambda_RClo, current_lambda_char = update_penalties(
                solution, current_lambda_size, current_lambda_RCup, current_lambda_RClo, current_lambda_char
            )
    final_cost = costs[-1]
    print(f" Final cost for restart {restart + 1}: {final_cost:.4f}")
    if final_cost < best_cost:
        best_cost = final_cost
        best_params = params.copy()

print(f"\nBest VQE cost: {best_cost:.4f}")

# 5. Sample and Post-process
print("\nSampling from optimized circuit...")
try:
    dev_sample = qml.device("lightning.qubit", wires=n, shots=50000)
    print("Using lightning.qubit for sampling")
except:
    dev_sample = qml.device("default.qubit", wires=n, shots=50000)
    print("Using default.qubit for sampling")

@qml.qnode(dev_sample)
def sample_circuit(params):
    qaoa_ansatz(params, wires=range(n))
    return [qml.sample(qml.PauliZ(w)) for w in range(n)]

samples = np.array(sample_circuit(best_params))
bit_strings = ((1 - samples) // 2).astype(int)
counts = Counter(map(tuple, bit_strings.T))

# Greedy Post-processing
def greedy_post_process(solution, N):
    sol = np.array(solution)
    if sol.sum() == N:
        return sol
    indices = np.argsort(classical_objective(sol))  # Sort by contribution to cost
    if sol.sum() > N:
        sol[indices[N:]] = 0  # Remove excess bonds
    else:
        sol[indices[:N - sol.sum()]] = 1  # Add bonds to reach N
    return sol

best_solution = None
best_quantum_cost = float('inf')
for solution, freq in counts.most_common(20):
    refined_solution = greedy_post_process(solution, N)
    cost = classical_objective(refined_solution)
    if cost < best_quantum_cost:
        best_quantum_cost = cost
        best_solution = refined_solution
        print(f" Feasible quantum solution: {[int(x) for x in refined_solution]} (cost: {cost:.2f}, freq: {freq})")

print(f"\nBest quantum solution: {[int(x) for x in best_solution]}")
print(f"Quantum cost: {best_quantum_cost:.4f}")

# 6. Classical Benchmark with Simulated Annealing
print("\nSolving classically with simulated annealing...")
def sa_objective(x):
    return classical_objective(np.round(x))  # Round to binary

x0 = np.random.randint(0, 2, n)
result = opt.basinhopping(
    sa_objective, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs={"method": "L-BFGS-B", "bounds": [(0, 1)] * n}
)
best_classical_solution = np.round(result.x).astype(int)
best_classical_cost = classical_objective(best_classical_solution)
print(f"Best classical solution: {[int(x) for x in best_classical_solution]}")
print(f"Classical cost: {best_classical_cost:.4f}")

# 7. Constraint Satisfaction Analysis
def analyze_solution(solution, name):
    sol = np.array(solution)
    basket_size = sol.sum()
    cash_flow = np.dot(a_cf, sol)
    characteristic = np.dot(char_coeff, sol)
    tracking_error = abs(np.dot(beta[:, 0] * x_c, sol) - k_target_lj)
    basket_violation = abs(basket_size - N)
    cash_violation = max(0, rc_min - cash_flow) + max(0, cash_flow - rc_max)
    char_violation = max(0, b_lo - characteristic) + max(0, characteristic - b_up)
    print(f"\n{name} Solution Analysis:")
    print(f" Portfolio: {[int(x) for x in solution]}")
    print(f" Basket size: {basket_size} (target: {N}) - Violation: {basket_violation}")
    print(f" Cash flow: {cash_flow:.4f} (range: [{rc_min:.4f}, {rc_max:.4f}]) - Violation: {cash_violation:.4f}")
    print(f" Characteristic: {characteristic:.4f} (range: [{b_lo:.4f}, {b_up:.4f}]) - Violation: {char_violation:.4f}")
    print(f" Tracking error: {tracking_error:.4f}")
    total_violation = basket_violation + cash_violation + char_violation
    print(f" Total constraint violation: {total_violation:.4f}")
    return total_violation

quantum_violation = analyze_solution(best_solution, "Quantum")
classical_violation = analyze_solution(best_classical_solution, "Classical")

# 8. Summary
print(f"\n=== Final Summary ===")
print(f"Problem size: {n} bonds, Target basket: {N}")
print(f"QAOA layers: {p}, Restarts: {num_restarts}, Iterations per restart: 500")
print(f"Best quantum cost: {best_quantum_cost:.4f} (violation: {quantum_violation:.4f})")
print(f"Best classical cost: {best_classical_cost:.4f} (violation: {classical_violation:.4f})")
print(f"Cost ratio (quantum/classical): {best_quantum_cost/best_classical_cost:.3f}")

# Calculate relative difference
relative_diff = abs(best_quantum_cost - best_classical_cost) / abs(best_classical_cost)
tolerance = 0.01  # 1% tolerance

if quantum_violation < classical_violation:
    print("QUANTUM ADVANTAGE ACHIEVED (better constraint satisfaction)")
elif best_quantum_cost < best_classical_cost:
    print("QUANTUM ADVANTAGE ACHIEVED (better objective value)")
elif relative_diff < tolerance:
    print(f"NEAR TIE - Difference: {relative_diff*100:.2f}% (within {tolerance*100:.0f}% tolerance)")
else:
    print("Classical solution dominates")

print(f"\nTop 5 quantum solutions:")
for i, (solution, freq) in enumerate(counts.most_common(5)):
    cost = classical_objective(np.array(solution))
    violation = sum([int(x) for x in solution]) != N
    print(f" {i+1}. {[int(x) for x in solution]} (freq: {freq:4d}, cost: {cost:7.2f}, feasible: {not violation})")
    
# Save top 5 quantum solutions data
top_quantum_solutions = []
for i, (solution, freq) in enumerate(counts.most_common(5)):
    cost = classical_objective(np.array(solution))
    top_quantum_solutions.append((cost, freq))
    
print("\nTop 5 Quantum Solutions Data for Plotting:")
for i, (cost, freq) in enumerate(top_quantum_solutions):
    print(f"Solution {i+1}: Cost = {cost:.2f}, Frequency = {freq}")