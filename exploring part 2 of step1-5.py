import pennylane.numpy as np
import pennylane as qml
from collections import Counter
import scipy.optimize as opt
import time
import gc
import torch

print("=== Exploring (part 2) of Quantum Portfolio Optimization... ===\n")

# 1. Optimized Problem Setup
n = 20  # Number of bonds
np.random.seed(42)

# Market Parameters
m = np.random.uniform(0.5, 1.0, size=n)
M = np.random.uniform(1.0, 2.0, size=n)
i_c = np.random.uniform(0.2, 0.8, size=n)
delta_c = np.random.uniform(0.1, 0.5, size=n)

# Global Parameters
N = 10  # Target basket size
rc_min = 0.01
rc_max = 0.05
mvb = 1.0

# Risk Characteristics
beta = np.random.uniform(0.0, 1.0, size=(n, 1))
k_target = np.array([[1.0]])
rho = np.array([[5.0]])

# Penalty Parameters
lambda_size = 2000.0
lambda_RCup = 500.0
lambda_RClo = 500.0
lambda_char = 300.0

# Pre-compute x_c
x_c = (m + np.minimum(M, i_c)) / (2 * delta_c)
print(f"\nProblem size: {n} bonds, Target basket: {N}")

print(f"Market parameters (m): {m}")
print()
print(f"Market parameters (M): {M}")
print()
print(f"Risk characteristics (i_c): {i_c}")
print()
print(f"Bond amounts if selected (x_c): {x_c}")
print()
print(f"Target basket size: {N}")
print()
print(f"Cash flow range: [{rc_min:.4f}, {rc_max:.4f}]")
print(f"Characteristic range: [0.6, 1.2]")


# 2. Build QUBO Formulation
Q = np.zeros((n, n))
q = np.zeros(n)
print("Building QUBO formulation...")

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

# 3. Create lightning-optimized device
def get_backend_device(n_qubits, shots=None):
    """Get the best optimized device available"""
    device_kwargs = {"wires": n_qubits}
    if shots is not None:
        device_kwargs["shots"] = shots
    
    # Try different backends in order of preference
    backend_to_try = [
        ("lightning.qubit", "Optimized CPU"),
    ]

    for backend, description in backend_to_try:
        try:
            device = qml.device(backend, **device_kwargs)
            print(f"Using {backend}: {description}")
            
            return device, backend
        except Exception as e:
            print(f"{backend} not available: {str(e)[:50]}...")
            continue
    
    # Should never reach here, but fallback just in case
    return qml.device("default.qubit", **device_kwargs), "default.qubit"

# 4. Optimized Hamiltonian Construction
print("Building Optimized Hamiltonian...")
coefficients = []
obs = []
constant = 0.0

# Add significant terms only (memory optimization)
for i in range(n):
    if abs(q[i]) > 1e-10:
        constant += q[i] / 2
        coefficients.append(-q[i] / 2)
        obs.append(qml.PauliZ(i))

for i in range(n):
    for j in range(i, n):
        coeff = Q[i, j]
        if abs(coeff) > 1e-10:
            if i == j:
                constant += coeff / 4
                coefficients.append(-coeff / 4)
                obs.append(qml.PauliZ(i))
            else:
                constant += coeff / 4
                coefficients.append(coeff / 4)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

obs.append(qml.Identity(0))
coefficients.append(constant)
Hamiltonian_cost = qml.Hamiltonian(coefficients, obs)
print(f"Hamiltonian constructed with {len(coefficients)} terms")

# 5. Optimized-Accelerated VQE
dev, device_name = get_backend_device(n)
print(f"Main computation device: {device_name}")

p = 4  # QAOA layers 

def lightning_optimized_qaoa(params, wires):
    """Optimized QAOA ansatz"""
    gammas = params[:p]
    betas = params[p:]
    
    # Initial superposition
    for wire in range(n):
        qml.Hadamard(wire)
    
    # QAOA layers with optimized gate application
    for layer in range(p):
        # Cost Hamiltonian - apply significant interactions only
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-8:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * gammas[layer] * Q[i, j], wires=j)
                    qml.CNOT(wires=[i, j])
        
        # Linear terms
        for i in range(n):
            if abs(q[i]) > 1e-8:
                qml.RZ(2 * gammas[layer] * q[i], wires=i)
        
        # Mixer Hamiltonian
        for i in range(n):
            qml.RX(2 * betas[layer], wires=i)

@qml.qnode(dev)
def circuit(params):
    lightning_optimized_qaoa(params, wires=range(n))
    return qml.expval(Hamiltonian_cost)

# 6. Optimized-Accelerated Training
print(f"\nStarting Optimized-Accelerated VQE training...")
start_training = time.time()

best_params = None
best_cost = float('inf')
num_restarts = 4  

for restart in range(num_restarts):
    print(f"\n--- Restart {restart + 1}/{num_restarts} ---")
    # Random initialization
    np.random.seed(42 + restart * 10)
    params = np.random.uniform(0, np.pi, 2 * p)
    
    # Use adaptive learning rate
    initial_stepsize = 0.05
    optimizer = qml.AdamOptimizer(stepsize=initial_stepsize)
    costs = []

    max_iterations = 200
    for it in range(max_iterations):
        params, cost = optimizer.step_and_cost(circuit, params)
        costs.append(cost)
        
        if it % 50 == 0:
            print(f" Iteration {it:3d}, Cost: {cost:8.4f}")
        
        # Adaptive learning rate
        if it > 0 and it % 100 == 0:
            if abs(costs[-1] - costs[-50]) < 1e-4:
                optimizer.stepsize *= 0.5  # Reduce learning rate
                print(f" Reduced learning rate to {optimizer.stepsize:.4f}")
        
        # Early stopping
        if it > 100 and abs(costs[-1] - costs[-20]) < 1e-6:
            print(f" Converged at iteration {it}")
            break
    
    final_cost = costs[-1]
    print(f" Final cost: {final_cost:.4f}")
    
    if final_cost < best_cost:
        best_cost = final_cost
        best_params = params.copy()

training_time = time.time() - start_training
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Best VQE cost: {best_cost:.4f}")

# 7. Optimized-Accelerated Sampling
print("\nOptimized-accelerated sampling...")
sampling_start = time.time()

dev_sample, sample_device = get_backend_device(n, shots=50000)
print(f"Sampling device: {sample_device}")

@qml.qnode(dev_sample)
def sample_circuit(params):
    lightning_optimized_qaoa(params, wires=range(n))
    return [qml.sample(qml.PauliZ(w)) for w in range(n)]

samples = np.array(sample_circuit(best_params))
sampling_time = time.time() - sampling_start
print(f"Sampling completed in {sampling_time:.2f} seconds")

bit_strings = ((1 - samples) // 2).astype(int)
counts = Counter(map(tuple, bit_strings.T))

# 8. Greedy Enhanced Post-processing
def enhanced_post_process(solution, N):
    """Enhanced post-processing with constraint optimization"""
    sol = np.array(solution, dtype=int)
    
    if sol.sum() == N:
        return sol
    
    # Calculate marginal contribution of each bond
    base_cost = classical_objective(sol)
    contributions = []
    
    for i in range(n):
        temp_sol = sol.copy()
        temp_sol[i] = 1 - temp_sol[i]  # Flip bit
        new_cost = classical_objective(temp_sol)
        contributions.append((new_cost - base_cost, i))
    
    contributions.sort()  # Sort by cost change
    
    # Adjust solution to meet size constraint
    if sol.sum() > N:
        # Remove worst bonds
        for cost_change, i in contributions:
            if sol[i] == 1 and sol.sum() > N:
                sol[i] = 0
    else:
        # Add best bonds
        for cost_change, i in contributions:
            if sol[i] == 0 and sol.sum() < N and cost_change < 0:
                sol[i] = 1
    
    return sol

# Find best quantum solution
best_solution = None
best_quantum_cost = float('inf')

print("\nTop quantum solutions (post-processed):")
for i, (solution, freq) in enumerate(counts.most_common(15)):
    refined_solution = enhanced_post_process(solution, N)
    cost = classical_objective(refined_solution)
    
    if cost < best_quantum_cost:
        best_quantum_cost = cost
        best_solution = refined_solution
    
    if i < 10:  # Show top 10
        print(f" {i+1:2d}. {[int(x) for x in refined_solution]} (cost: {cost:8.2f}, freq: {freq:4d})")

# 9. Classical Benchmark
print("\nClassical benchmark...")
classical_start = time.time()

def sa_objective(x):
    return classical_objective(np.round(x))

x0 = np.random.randint(0, 2, n)
result = opt.basinhopping(
    sa_objective, x0, niter=100, T=1.0, stepsize=0.5,
    minimizer_kwargs={"method": "L-BFGS-B", "bounds": [(0, 1)] * n}
)
classical_time = time.time() - classical_start

best_classical_solution = np.round(result.x).astype(int)
best_classical_cost = classical_objective(best_classical_solution)

# 10. Performance Analysis
print(f"\n=== Optimized Performance Summary (with lightning) ===")
print(f"Hardware: {device_name}")

print(f"Problem size: {n} bonds, Target basket: {N}")
print(f"QAOA layers: {p}, Restarts: {num_restarts}")
print(f"\nTiming:")
print(f"  Training: {training_time:.2f}s")
print(f"  Sampling: {sampling_time:.2f}s") 
print(f"  Classical: {classical_time:.2f}s")
print(f"  Total quantum: {training_time + sampling_time:.2f}s")

print(f"\nResults:")
print(f"  Best quantum cost: {best_quantum_cost:.4f}")
print(f"  Best classical cost: {best_classical_cost:.4f}")
print(f"  Quantum advantage: {best_classical_cost/best_quantum_cost:.3f}x")

# Solution analysis
def analyze_solution(solution, name):
    sol = np.array(solution)
    basket_size = sol.sum()
    cash_flow = np.dot(a_cf, sol)
    characteristic = np.dot(char_coeff, sol)
    
    print(f"\n{name} Solution:")
    print(f"  Portfolio: {[int(x) for x in solution]}")
    print(f"  Basket size: {basket_size} (target: {N})")
    print(f"  Cash flow: {cash_flow:.4f} (range: [{rc_min:.4f}, {rc_max:.4f}])")
    print(f"  Characteristic: {characteristic:.4f} (range: [{b_lo:.4f}, {b_up:.4f}])")
    
    # Constraint violations
    size_violation = abs(basket_size - N)
    cash_violation = max(0, rc_min - cash_flow) + max(0, cash_flow - rc_max)
    char_violation = max(0, b_lo - characteristic) + max(0, characteristic - b_up)
    total_violation = size_violation + cash_violation + char_violation
    print(f"  Constraint violation: {total_violation:.4f}")
    
    return total_violation

quantum_violation = analyze_solution(best_solution, "Quantum")
classical_violation = analyze_solution(best_classical_solution, "Classical")

# Final assessment
print(f"\n=== Final Assessment ===")
if quantum_violation < classical_violation:
    print("QUANTUM ADVANTAGE: Better constraint satisfaction")
elif best_quantum_cost < best_classical_cost * 0.99:
    print("QUANTUM ADVANTAGE: Better objective value")
elif abs(best_quantum_cost - best_classical_cost) / best_classical_cost < 0.02:
    print("COMPETITIVE: Near-optimal performance")
else:
    print("Classical solution dominates")
    
# Save top 10 quantum solutions data
top_quantum_solutions = []
for i, (solution, freq) in enumerate(counts.most_common(10)):
    cost = classical_objective(np.array(solution))
    top_quantum_solutions.append((cost, freq))

print("\nTop 10 Quantum Solutions Data for Plotting:")
for i, (cost, freq) in enumerate(top_quantum_solutions):
    print(f"Solution {i+1}: Cost = {cost:.2f}, Frequency = {freq}")