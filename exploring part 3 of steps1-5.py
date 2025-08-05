import pennylane.numpy as np
import pennylane as qml
from collections import Counter
import scipy.optimize as opt
import time
import gc
import torch
import psutil
import threading
from multiprocessing import Pool, Manager
import hashlib

print("=== Hardware-Optimized Quantum Portfolio Optimization ===\n")

# 1. Hardware-Specific Backend Selection
def setup_optimized_backend(n_qubits, shots=None):
    """Setup the best available backend with GPU support"""
    device_kwargs = {"wires": n_qubits}
    if shots is not None:
        device_kwargs["shots"] = shots
    
    # Try GPU first
    try:
        import cupy as cp
        gpu_count = cp.cuda.runtime.getDeviceCount()
        if gpu_count > 0:
            try:
                device = qml.device('lightning.gpu', **device_kwargs)
                print(f"Using GPU acceleration with {gpu_count} GPU(s)")
                return device, "lightning.gpu"
            except Exception as e:
                print(f"GPU backend failed: {str(e)[:50]}...")
    except ImportError:
        print("GPU libraries not available")
    
    # Fallback to optimized CPU
    try:
        device = qml.device('lightning.qubit', **device_kwargs)
        print("Using optimized CPU backend (Lightning)")
        return device, "lightning.qubit"
    except:
        device = qml.device('default.qubit', **device_kwargs)
        print("Using default backend (slower)")
        return device, "default.qubit"

# 2. Memory-Optimized Circuit Construction
class OptimizedCircuitBuilder:
    def __init__(self, n_qubits, threshold=1e-8):
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.significant_gates = []
        self.memory_estimate = 0
        
    def analyze_hamiltonian(self, Q, q):
        """Analyze and compile significant Hamiltonian terms"""
        self.significant_gates = []
        
        # Single qubit terms
        for i in range(self.n_qubits):
            if abs(q[i]) > self.threshold:
                self.significant_gates.append(('single', i, q[i]))
        
        # Two qubit terms
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                if abs(Q[i, j]) > self.threshold:
                    self.significant_gates.append(('pair', (i, j), Q[i, j]))
        
        # Memory estimation
        state_memory = 2**self.n_qubits * 16  # Complex128
        gate_memory = len(self.significant_gates) * 64
        self.memory_estimate = (state_memory + gate_memory) / (1024**2)
        
        print(f"Circuit Analysis:")
        print(f"  Significant gates: {len(self.significant_gates)}")
        print(f"  Estimated memory: {self.memory_estimate:.2f} MB")
        
        return len(self.significant_gates)
    
    def apply_cost_layer(self, gamma):
        """Apply optimized cost layer"""
        for gate_type, qubits, coeff in self.significant_gates:
            if gate_type == 'single':
                qml.RZ(2 * gamma * coeff, wires=qubits)
            elif gate_type == 'pair':
                i, j = qubits
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * coeff, wires=j)
                qml.CNOT(wires=[i, j])

# 3. Advanced Caching System
class SmartCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _param_hash(self, params, precision=6):
        """Create consistent hash for parameters"""
        rounded = np.round(params, precision)
        return hashlib.md5(rounded.tobytes()).hexdigest()
    
    def get_or_compute(self, params, compute_func):
        """Get cached result or compute and cache"""
        key = self._param_hash(params)
        
        if key in self.cache:
            self.hits += 1
            self.access_count[key] += 1
            return self.cache[key]
        
        # Cache miss - compute result
        self.misses += 1
        result = compute_func(params)
        
        # Manage cache size
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        # Store new result
        self.cache[key] = result
        self.access_count[key] = 1
        
        return result
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {len(self.cache)}/{self.max_size} entries, {hit_rate:.1f}% hit rate"

# 4. Real-time Performance Monitor
class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.monitoring = False
        self.start_time = None
    
    def start(self):
        """Start monitoring in background thread"""
        self.start_time = time.time()
        self.monitoring = True
        self.metrics = []
        
        def monitor_loop():
            while self.monitoring:
                try:
                    process = psutil.Process()
                    self.metrics.append({
                        'time': time.time() - self.start_time,
                        'cpu': process.cpu_percent(),
                        'memory_mb': process.memory_info().rss / 1024**2
                    })
                    time.sleep(1)  # Monitor every second
                except:
                    break
        
        self.thread = threading.Thread(target=monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return summary"""
        self.monitoring = False
        if not self.metrics:
            return "No monitoring data"
        
        cpu_vals = [m['cpu'] for m in self.metrics]
        mem_vals = [m['memory_mb'] for m in self.metrics]
        
        return {
            'duration': self.metrics[-1]['time'],
            'avg_cpu': np.mean(cpu_vals),
            'max_cpu': np.max(cpu_vals),
            'avg_memory': np.mean(mem_vals),
            'max_memory': np.max(mem_vals),
            'peak_memory': np.max(mem_vals)
        }

# 5. Adaptive Resource Manager
class ResourceManager:
    @staticmethod
    def get_optimal_config(n_qubits):
        """Get optimal configuration based on hardware"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        config = {
            'qaoa_layers': min(6, max(2, int(memory_gb // 2))),
            'shots': min(100000, int(memory_gb * 5000)),
            'restarts': min(cpu_count, max(2, n_qubits // 3)),
            'processes': min(cpu_count - 1, 4),
            'batch_size': max(100, int(10000 / (2**max(0, n_qubits-10))))
        }
        
        print(f"Hardware Config:")
        print(f"  CPUs: {cpu_count}, Memory: {memory_gb:.1f}GB")
        print(f"  Recommended: {config}")
        
        return config

# 6. Enhanced Problem Setup (same as original but with optimizations)
n = 20
np.random.seed(42)

# Market Parameters (same as original)
m = np.random.uniform(0.5, 1.0, size=n)
M = np.random.uniform(1.0, 2.0, size=n)
i_c = np.random.uniform(0.2, 0.8, size=n)
delta_c = np.random.uniform(0.1, 0.5, size=n)

N = 10
rc_min = 0.01
rc_max = 0.07
mvb = 1.0

beta = np.random.uniform(0.0, 1.0, size=(n, 1))
k_target = np.array([[1.0]])
rho = np.array([[5.0]])

lambda_size = 2000.0
lambda_RCup = 500.0
lambda_RClo = 500.0
lambda_char = 300.0

x_c = (m + np.minimum(M, i_c)) / (2 * delta_c)

print(f"Problem: {n} bonds → {N} portfolio")

# 7. QUBO Construction (same as original)
Q = np.zeros((n, n))
q = np.zeros(n)

# Build QUBO (same logic as original)
l, j = 0, 0
w = rho[l, j]
k_target_lj = k_target[l, j]

for i in range(n):
    for k in range(n):
        Q[i, k] += w * beta[i, j] * beta[k, j] * x_c[i] * x_c[k]
for i in range(n):
    q[i] += -2 * w * beta[i, j] * x_c[i] * k_target_lj

Q += lambda_size * np.ones((n, n))
np.fill_diagonal(Q, np.diag(Q) - lambda_size)
q += -2 * lambda_size * N * np.ones(n)

a_cf = (m * delta_c * x_c) / (100 * mvb)
Q += lambda_RCup * np.outer(a_cf, a_cf)
q += -2 * lambda_RCup * rc_max * a_cf
Q += lambda_RClo * np.outer(a_cf, a_cf)
q += -2 * lambda_RClo * rc_min * a_cf

b_up = 1.6
b_lo = 0.6
char_coeff = beta[:, j] * i_c
Q += lambda_char * np.outer(char_coeff, char_coeff)
q += -2 * lambda_char * b_up * char_coeff
Q += lambda_char * np.outer(char_coeff, char_coeff)
q += -2 * lambda_char * b_lo * char_coeff

Q = (Q + Q.T) / 2

def classical_objective(x):
    return x.T @ Q @ x + q.T @ x

# 8. Initialize Optimization Components
print("\n=== Initializing Optimized Components ===")

# Get hardware configuration
config = ResourceManager.get_optimal_config(n)
p = config['qaoa_layers']

# Setup optimized backend
dev, backend_name = setup_optimized_backend(n)
print(f"Main device: {backend_name}")

# Initialize circuit optimizer
circuit_builder = OptimizedCircuitBuilder(n)
n_gates = circuit_builder.analyze_hamiltonian(Q, q)

# Initialize cache and monitor
cache = SmartCache(max_size=500)
monitor = PerformanceMonitor()

# 9. Optimized QAOA Circuit
def optimized_qaoa_ansatz(params, wires):
    """Hardware-optimized QAOA ansatz"""
    gammas = params[:p]
    betas = params[p:]
    
    # Initial superposition
    for wire in range(n):
        qml.Hadamard(wire)
    
    # QAOA layers with optimized gates
    for layer in range(p):
        # Apply cost layer using pre-compiled gates
        circuit_builder.apply_cost_layer(gammas[layer])
        
        # Mixer layer
        for wire in range(n):
            qml.RX(2 * betas[layer], wires=wire) # Maintains superposition

@qml.qnode(dev, diff_method="adjoint")
def optimized_circuit(params):
    optimized_qaoa_ansatz(params, wires=range(n))
    return qml.expval(qml.Hamiltonian(
        [1.0] + [-q[i]/2 for i in range(n)] + [Q[i,j]/4 for i in range(n) for j in range(i+1,n) if abs(Q[i,j]) > 1e-10],
        
        [qml.Identity(0)] + [qml.PauliZ(i) for i in range(n)] + [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(n) for j in range(i+1,n) if abs(Q[i,j]) > 1e-10]
    ))

# 10. Optimized Training with Caching
print(f"\n=== Starting Optimized Training ===")
print(f"QAOA layers: {p}, Restarts: {config['restarts']}")

# Start performance monitoring
monitor.start()

def cached_objective(params):
    """Cached version of circuit evaluation"""
    return cache.get_or_compute(params, optimized_circuit)

best_params = None
best_cost = float('inf')
training_start = time.time()

for restart in range(config['restarts']):
    print(f"\n--- Restart {restart + 1}/{config['restarts']} ---")
    
    # Smart initialization
    if restart == 0:
        params = np.random.uniform(0, np.pi/2, 2 * p)  # Conservative
    elif restart == 1:
        params = np.random.uniform(0, np.pi, 2 * p)    # Standard
    else:
        params = np.random.uniform(0, 2*np.pi, 2 * p)  # Aggressive
    
    optimizer = qml.AdamOptimizer(stepsize=0.05)
    costs = []
    
    for it in range(200):
        params, cost = optimizer.step_and_cost(cached_objective, params)
        costs.append(cost)
        
        if it % 50 == 0:
            print(f"  Iter {it:3d}: {cost:8.4f} | {cache.stats()}")
        
        # Adaptive learning rate
        if it > 0 and it % 100 == 0:
            if len(costs) > 50 and abs(costs[-1] - costs[-50]) < 1e-4:
                optimizer.stepsize *= 0.7
        
        # Early stopping
        if it > 50 and len(costs) > 20 and abs(costs[-1] - costs[-20]) < 1e-6:
            print(f"  Converged at iteration {it}")
            break
    
    final_cost = costs[-1]
    if final_cost < best_cost:
        best_cost = final_cost
        best_params = params.copy()
        print(f"New best: {final_cost:.4f}")

training_time = time.time() - training_start

# 11. Optimized Sampling
print(f"\n=== Optimized Sampling ===")
sampling_start = time.time()

# Use separate device for sampling
dev_sample, _ = setup_optimized_backend(n, shots=config['shots'])

@qml.qnode(dev_sample)
def sample_circuit(params):
    optimized_qaoa_ansatz(params, wires=range(n))
    return [qml.sample(qml.PauliZ(w)) for w in range(n)]

samples = np.array(sample_circuit(best_params))
sampling_time = time.time() - sampling_start

# Process samples
bit_strings = ((1 - samples) // 2).astype(int)
counts = Counter(map(tuple, bit_strings.T))

print(f"Sampling: {config['shots']} shots in {sampling_time:.2f}s")

# 12. Enhanced Post-processing (same as original)
def enhanced_post_process(solution, N):
    sol = np.array(solution, dtype=int)
    if sol.sum() == N:
        return sol
    
    base_cost = classical_objective(sol)
    contributions = []
    
    for i in range(n):
        temp_sol = sol.copy()
        temp_sol[i] = 1 - temp_sol[i]
        new_cost = classical_objective(temp_sol)
        contributions.append((new_cost - base_cost, i))
    
    contributions.sort()
    
    if sol.sum() > N:
        for cost_change, i in contributions:
            if sol[i] == 1 and sol.sum() > N:
                sol[i] = 0
    else:
        for cost_change, i in contributions:
            if sol[i] == 0 and sol.sum() < N and cost_change < 0:
                sol[i] = 1
    
    return sol

# Find best quantum solution
best_solution = None
best_quantum_cost = float('inf')

print(f"\nTop quantum solutions:")
for i, (solution, freq) in enumerate(counts.most_common(10)):
    refined_solution = enhanced_post_process(solution, N)
    cost = classical_objective(refined_solution)
    
    if cost < best_quantum_cost:
        best_quantum_cost = cost
        best_solution = refined_solution
    
    print(f"  {i+1:2d}. Cost: {cost:8.2f}, Freq: {freq:4d}")

# 13. Performance Summary
performance_summary = monitor.stop()
print(f"\n=== Hardware-Optimized Performance Summary ===")
print(f"Backend: {backend_name}")
print(f"Circuit gates: {n_gates} (optimized)")
print(f"Memory usage: {circuit_builder.memory_estimate:.2f} MB")

print(f"\nTiming:")
print(f"  Training: {training_time:.2f}s")
print(f"  Sampling: {sampling_time:.2f}s")
print(f"  Total: {training_time + sampling_time:.2f}s")

print(f"\nSystem Performance:")
if isinstance(performance_summary, dict):
    print(f"  Peak CPU: {performance_summary['max_cpu']:.1f}%")
    print(f"  Peak Memory: {performance_summary['peak_memory']:.1f} MB")
    print(f"  Duration: {performance_summary['duration']:.1f}s")

print(f"\nOptimization Stats:")
print(f"  {cache.stats()}")
print(f"  Best quantum cost: {best_quantum_cost:.4f}")

# 14. Solution Analysis (same as original)
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
    
    size_violation = abs(basket_size - N)
    cash_violation = max(0, rc_min - cash_flow) + max(0, cash_flow - rc_max)
    char_violation = max(0, b_lo - characteristic) + max(0, characteristic - b_up)
    total_violation = size_violation + cash_violation + char_violation
    print(f"  Constraint violation: {total_violation:.4f}")
    
    return total_violation

analyze_solution(best_solution, "Optimized Quantum")

print(f"\n=== Optimization Impact ===")
print(f"Hardware-optimized backend: {backend_name}")
print(f"Memory-optimized circuits: {n_gates} gates")
print(f"Smart caching: {cache.stats()}")
print(f"Real-time monitoring: Peak {performance_summary.get('peak_memory', 0):.0f}MB")
print(f"Adaptive configuration: {config['qaoa_layers']} layers, {config['restarts']} restarts")

# Clean up
gc.collect()
print("\nMemory cleanup completed")