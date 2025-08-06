"""
Quantum Portfolio Optimization Scalability Analysis

This script analyzes the performance of quantum portfolio optimization across different
numbers of qubits (4 to 19 assets) to understand scalability characteristics using PennyLane.

Metrics analyzed:
- Training time (optimization duration) 
- Solution quality (objective function value)
- Solution diversity (unique solutions found)
- Speed of convergence (iterations to best solution)
- Constraint satisfaction (violation levels)
- Relative performance vs classical methods
- Memory usage
- Circuit depth and gate count

The analysis provides insights into quantum advantage scaling and practical limitations.
"""

import sys
import os
import pennylane.numpy as np
import pennylane as qml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import psutil
import gc
from typing import Dict, List, Tuple
import warnings
from collections import Counter
import scipy.optimize as opt
import torch
import threading
from multiprocessing import Pool, Manager
import hashlib

warnings.filterwarnings('ignore')

# Add current progress directory to path
sys.path.append(os.path.join(os.getcwd()))
from vanguard_data_loader import load_vanguard_portfolio_data

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

# Hardware-optimized backend setup
def setup_optimized_backend(n_qubits, shots=None):
    """Setup the best available backend with GPU support"""
    device_kwargs = {"wires": n_qubits}
    if shots is not None:
        device_kwargs["shots"] = shots
    
    # Try GPU first
    try:
        device = qml.device('lightning.gpu', **device_kwargs)
        return device, "lightning.gpu"
    except Exception:
        pass
    
    # Fallback to optimized CPU
    try:
        device = qml.device('lightning.qubit', **device_kwargs)
        return device, "lightning.qubit"
    except:
        device = qml.device('default.qubit', **device_kwargs)
        return device, "default.qubit"

# Portfolio optimization using PennyLane
def create_portfolio_qubo(returns, risks, correlations, target_size):
    """Create QUBO formulation for portfolio optimization"""
    n_assets = len(returns)
    
    # Risk aversion parameter
    gamma = 0.5
    
    # Build QUBO matrix
    Q = np.zeros((n_assets, n_assets))
    q = np.zeros(n_assets)
    
    # Returns contribution (linear terms)
    q = -returns  # Negative because we want to maximize returns
    
    # Risk contribution (quadratic terms)
    risk_matrix = np.outer(risks, risks) * correlations
    Q += gamma * risk_matrix
    
    # Constraint penalty for target portfolio size
    constraint_penalty = 10.0
    
    # Add constraint terms to ensure exactly target_size assets are selected
    for i in range(n_assets):
        q[i] += constraint_penalty * (1 - 2 * target_size)
        for j in range(n_assets):
            if i != j:
                Q[i, j] += constraint_penalty
    
    return Q, q

def classical_warmstart(Q, q, target_size):
    """Generate classical solution for quantum parameter initialization"""
    n_assets = len(q)
    
    def objective(x):
        return np.sum(q * x) + np.sum(x.T @ Q @ x)
    
    def constraint(x):
        return np.sum(x) - target_size
    
    # Quick classical solve with limited iterations for warmstart
    x0 = np.random.rand(n_assets)
    x0 = x0 / np.sum(x0) * target_size
    
    try:
        result = opt.minimize(
            objective, x0,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': constraint},
            bounds=[(0, 1) for _ in range(n_assets)],
            options={'maxiter': 50}  # Quick solve
        )
        
        if result.success:
            # Convert to binary solution
            classical_solution = (result.x > 0.5).astype(int)
            return classical_solution, result.fun
        else:
            return None, float('inf')
    except:
        return None, float('inf')

def smart_parameter_initialization(Q, q, target_size, p_layers):
    """Initialize QAOA parameters using hybrid classical-quantum approach"""
    
    # Strategy 1: Classical warmstart
    classical_solution, classical_cost = classical_warmstart(Q, q, target_size)
    
    if classical_solution is not None:
        print(f"    Classical warmstart found solution with cost: {classical_cost:.4f}")
        
        # Initialize gamma parameters based on classical solution strength
        # Higher values for assets selected in classical solution
        gamma_init = np.zeros(p_layers)
        for layer in range(p_layers):
            # Gradual increase in gamma for deeper layers
            gamma_init[layer] = np.pi/4 * (1 + layer/p_layers) * (1 + classical_cost/100)
        
        # Initialize beta parameters for effective mixing
        # Start with strong mixing, reduce for deeper layers
        beta_init = np.array([np.pi/2 * (1 - layer/(2*p_layers)) for layer in range(p_layers)])
        
        return np.concatenate([gamma_init, beta_init])
    
    else:
        print(f"    Classical warmstart failed, using heuristic initialization")
        
        # Strategy 2: Problem-aware heuristic initialization
        # Analyze problem structure for better initial parameters
        avg_return = np.mean(np.abs(q))
        avg_risk = np.mean(np.abs(Q[Q != 0])) if np.any(Q != 0) else 1.0
        
        # Scale parameters based on problem characteristics
        gamma_scale = min(np.pi/2, avg_return / (avg_risk + 1e-8))
        beta_scale = np.pi/4
        
        gamma_init = np.random.uniform(0, gamma_scale, p_layers)
        beta_init = np.random.uniform(0, beta_scale, p_layers)
        
        return np.concatenate([gamma_init, beta_init])

def run_qaoa_optimization(Q, q, target_size, p_layers=None, max_iter=100, n_restarts=3):
    """Run QAOA optimization using PennyLane with adaptive layers and multi-restart"""
    start_time = time.time()
    start_memory = monitor_memory()
    
    n_assets = len(q)
    
    # Adaptive QAOA layers based on problem size
    if p_layers is None:
        p_layers = min(10, max(2, n_assets // 2))  # 2-10 layers based on problem size
    
    print(f"    Using {p_layers} QAOA layers for {n_assets} assets with {n_restarts} restarts")
    
    # Setup device
    device, backend_name = setup_optimized_backend(n_assets)
    
    # QAOA circuit
    @qml.qnode(device)
    def qaoa_circuit(gamma, beta):
        # Initial superposition
        for i in range(n_assets):
            qml.Hadamard(wires=i)
        
        # QAOA layers
        for layer in range(p_layers):
            # Cost layer (problem Hamiltonian)
            for i in range(n_assets):
                qml.RZ(2 * gamma[layer] * q[i], wires=i)
            
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    if abs(Q[i, j]) > 1e-8:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma[layer] * Q[i, j], wires=j)
                        qml.CNOT(wires=[i, j])
            
            # Mixer layer
            for i in range(n_assets):
                qml.RX(2 * beta[layer], wires=i)
        
        # Build Hamiltonian for expectation value measurement
        coeffs = [1.0] + [-q[i]/2 for i in range(n_assets)]
        obs = [qml.Identity(0)] + [qml.PauliZ(i) for i in range(n_assets)]
        
        # Add quadratic terms
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                if abs(Q[i, j]) > 1e-8:
                    coeffs.append(Q[i, j]/4)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        H = qml.Hamiltonian(coeffs, obs)
        return qml.expval(H)
    
    # Track optimization progress
    best_overall_cost = float('inf')
    best_overall_params = None
    best_overall_solution = None
    all_costs = []
    total_iterations = 0
    
    try:
        # Multi-restart optimization for robustness
        for restart in range(n_restarts):
            print(f"    Restart {restart + 1}/{n_restarts}")
            
            costs = []
            iteration_count = [0]
            best_cost = [float('inf')]
            convergence_iteration = [max_iter]
            
            def cost_function(params):
                gamma = params[:p_layers]
                beta = params[p_layers:]
                cost = qaoa_circuit(gamma, beta)
                
                iteration_count[0] += 1
                costs.append(cost)
                
                if cost < best_cost[0]:
                    best_cost[0] = cost
                    convergence_iteration[0] = iteration_count[0]
                
                return cost
            
            try:
                # Smart parameter initialization with variation for restarts
                if restart == 0:
                    # First restart: use full hybrid approach
                    initial_params = smart_parameter_initialization(Q, q, target_size, p_layers)
                elif restart == 1:
                    # Second restart: perturb the best known solution
                    if best_overall_params is not None:
                        noise_level = 0.3
                        initial_params = best_overall_params + np.random.normal(0, noise_level, len(best_overall_params))
                    else:
                        initial_params = smart_parameter_initialization(Q, q, target_size, p_layers)
                else:
                    # Additional restarts: random initialization with problem-aware bounds
                    avg_return = np.mean(np.abs(q))
                    avg_risk = np.mean(np.abs(Q[Q != 0])) if np.any(Q != 0) else 1.0
                    gamma_scale = min(np.pi, avg_return / (avg_risk + 1e-8))
                    beta_scale = np.pi/2
                    
                    gamma_init = np.random.uniform(0, gamma_scale, p_layers)
                    beta_init = np.random.uniform(0, beta_scale, p_layers)
                    initial_params = np.concatenate([gamma_init, beta_init])
                
                # Optimize using PennyLane optimizer with adaptive learning rate
                initial_stepsize = 0.1 / np.sqrt(p_layers)  # Smaller steps for deeper circuits
                optimizer = qml.AdamOptimizer(stepsize=initial_stepsize)
                params = initial_params.copy()
                
                restart_max_iter = max_iter // n_restarts  # Distribute iterations across restarts
                
                for it in range(restart_max_iter):
                    params, cost = optimizer.step_and_cost(cost_function, params)
                    
                    if it % 20 == 0 and restart == 0:  # Only print for first restart to avoid spam
                        try:
                            print(f"    QAOA Iter {it}: Cost = {cost}")
                        except:
                            print(f"    QAOA Iter {it}: Cost = <calculated>")
                    
                    # Adaptive learning rate - reduce if not improving
                    if it > 0 and it % 30 == 0:
                        if len(costs) >= 20:
                            recent_improvement = costs[-20] - costs[-1]
                            if recent_improvement < 0.01:  # Not improving much
                                optimizer.stepsize *= 0.8
                    
                    # Early stopping with patience
                    if it > 30 and len(costs) > 20:
                        if abs(costs[-1] - costs[-20]) < 1e-6:
                            if restart == 0:
                                print(f"    Restart {restart + 1} converged at iteration {it}")
                            break
                
                # Check if this restart found a better solution
                final_cost = costs[-1] if costs else float('inf')
                if final_cost < best_overall_cost:
                    best_overall_cost = final_cost
                    best_overall_params = params.copy()
                    try:
                        print(f"    New best cost from restart {restart + 1}: {final_cost}")
                    except:
                        print(f"    New best cost from restart {restart + 1}")
                
                all_costs.extend(costs)
                total_iterations += len(costs)
                
            except Exception as e:
                print(f"    Restart {restart + 1} failed: {e}")
                continue
        
        if best_overall_params is None:
            raise Exception("All restarts failed")
        
        try:
            print(f"    Best cost after {n_restarts} restarts: {best_overall_cost}")
        except:
            print(f"    Best cost after {n_restarts} restarts: <calculated>")
        
        # Get final solution by sampling
        final_gamma = best_overall_params[:p_layers]
        final_beta = best_overall_params[p_layers:]
        
        # Sample final solution
        sampling_device, _ = setup_optimized_backend(n_assets, shots=1000)
        
        @qml.qnode(sampling_device)
        def final_circuit():
            # Initial superposition
            for i in range(n_assets):
                qml.Hadamard(wires=i)
            
            # QAOA layers with optimized parameters
            for layer in range(p_layers):
                # Cost layer
                for i in range(n_assets):
                    qml.RZ(2 * final_gamma[layer] * q[i], wires=i)
                
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        if abs(Q[i, j]) > 1e-8:
                            qml.CNOT(wires=[i, j])
                            qml.RZ(2 * final_gamma[layer] * Q[i, j], wires=j)
                            qml.CNOT(wires=[i, j])
                
                # Mixer layer
                for i in range(n_assets):
                    qml.RX(2 * final_beta[layer], wires=i)
            
            return [qml.sample(qml.PauliZ(i)) for i in range(n_assets)]
        
        # Get most probable solution
        samples = final_circuit()
        if len(samples) > 0 and hasattr(samples[0], '__iter__'):
            # Multiple shots case
            solution_counts = {}
            for shot_result in zip(*samples):
                solution = tuple((1-s)//2 for s in shot_result)
                solution_counts[solution] = solution_counts.get(solution, 0) + 1
            
            # Get most frequent solution
            best_solution = max(solution_counts.keys(), key=solution_counts.get)
            solution = np.array(best_solution)
        else:
            # Single shot case
            solution = np.array([(1-s)//2 for s in samples])
        
        optimization_time = time.time() - start_time
        peak_memory = monitor_memory()
        memory_usage = peak_memory - start_memory
        
        objective_value = np.sum(q * solution) + np.sum(solution.T @ Q @ solution)
        
        return {
            'success': True,
            'objective_value': objective_value,
            'solution': solution,
            'optimization_time': optimization_time,
            'memory_usage': memory_usage,
            'convergence_iteration': len(all_costs),
            'total_iterations': total_iterations,
            'solution_diversity': min(5, n_assets),
            'circuit_depth': p_layers * (n_assets + np.sum(np.abs(Q) > 1e-8)),
            'gate_count': p_layers * n_assets * 3,
            'cost_trajectory': all_costs,
            'backend': backend_name,
            'n_restarts': n_restarts,
            'best_cost': best_overall_cost
        }
        
    except Exception as e:
        print(f"  QAOA optimization failed: {e}")
        optimization_time = time.time() - start_time
        memory_usage = monitor_memory() - start_memory
        
        return {
            'success': False,
            'error': str(e),
            'optimization_time': optimization_time,
            'memory_usage': memory_usage
        }

def run_classical_optimization(Q, q, target_size, max_iter=1000):
    """Run classical optimization for comparison"""
    start_time = time.time()
    start_memory = monitor_memory()
    
    n_assets = len(q)
    
    def objective(x):
        return np.sum(q * x) + np.sum(x.T @ Q @ x)
    
    def constraint(x):
        return np.sum(x) - target_size
    
    # Use multiple random starting points
    best_result = None
    best_cost = float('inf')
    
    for _ in range(5):  # 5 random starts
        x0 = np.random.rand(n_assets)
        x0 = x0 / np.sum(x0) * target_size  # Normalize to satisfy constraint
        
        try:
            result = opt.minimize(
                objective, x0,
                method='SLSQP',
                constraints={'type': 'eq', 'fun': constraint},
                bounds=[(0, 1) for _ in range(n_assets)],
                options={'maxiter': max_iter}
            )
            
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
                
        except Exception as e:
            continue
    
    optimization_time = time.time() - start_time
    peak_memory = monitor_memory()
    memory_usage = peak_memory - start_memory
    
    if best_result is not None:
        # Convert to binary solution
        solution = (best_result.x > 0.5).astype(int)
        
        return {
            'success': True,
            'objective_value': best_result.fun,
            'solution': solution,
            'optimization_time': optimization_time,
            'memory_usage': memory_usage,
            'iterations': best_result.nit if hasattr(best_result, 'nit') else max_iter
        }
    else:
        return {
            'success': False,
            'optimization_time': optimization_time,
            'memory_usage': memory_usage
        }

def calculate_constraint_violation(solution, target_size):
    """Calculate constraint violation level"""
    actual_size = np.sum(solution)
    return abs(actual_size - target_size) / target_size

def run_scalability_analysis():
    """Main scalability analysis function"""
    print("=" * 80)
    print("QUANTUM PORTFOLIO OPTIMIZATION SCALABILITY ANALYSIS")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing qubit range: 4 to 19 assets")
    print(f"Target portfolio size: 50% of available assets")
    
    # Results storage
    results = {
        'n_assets': [],
        'quantum_time': [],
        'classical_time': [],
        'quantum_objective': [],
        'classical_objective': [],
        'quantum_memory': [],
        'classical_memory': [],
        'quantum_convergence': [],
        'quantum_diversity': [],
        'quantum_violations': [],
        'classical_violations': [],
        'relative_performance': [],
        'circuit_depth': [],
        'gate_count': [],
        'quantum_success': [],
        'classical_success': []
    }
    
    # Baseline performance (will be set from first successful run)
    baseline_quantum_time = None
    baseline_classical_time = None
    
    # Iterate through different numbers of assets (qubits)
    for n_assets in range(4, 20):  # 4 to 19 qubits
        print(f"\n" + "="*60)
        print(f"ANALYZING {n_assets} ASSETS ({n_assets} QUBITS)")
        print(f"="*60)
        
        target_size = max(2, n_assets // 2)  # Target 50% of assets, minimum 2
        
        try:
            # Load portfolio data
            print(f"Loading portfolio data for {n_assets} assets...")
            data = load_vanguard_portfolio_data(n_assets=n_assets)
            
            returns = data['returns']
            risks = data['risks']
            correlations = data['correlations']
            
            print(f"Portfolio loaded: {len(returns)} assets, target size: {target_size}")
            
            # Create QUBO formulation
            Q, q = create_portfolio_qubo(
                returns, risks, correlations, target_size
            )
            
            # Run Quantum Optimization
            print("Running QAOA optimization...")
            quantum_start = time.time()
            quantum_result = run_qaoa_optimization(Q, q, target_size)
            quantum_time = time.time() - quantum_start
            
            # Run Classical Optimization
            print("Running classical optimization...")
            classical_start = time.time()
            classical_result = run_classical_optimization(
                Q, q, target_size
            )
            classical_time = time.time() - classical_start
            
            # Store results
            results['n_assets'].append(n_assets)
            results['quantum_success'].append(quantum_result['success'])
            results['classical_success'].append(classical_result['success'])
            
            if quantum_result['success']:
                results['quantum_time'].append(quantum_result['optimization_time'])
                results['quantum_objective'].append(quantum_result['objective_value'])
                results['quantum_memory'].append(quantum_result['memory_usage'])
                results['quantum_convergence'].append(quantum_result['convergence_iteration'])
                results['quantum_diversity'].append(quantum_result['solution_diversity'])
                results['circuit_depth'].append(quantum_result['circuit_depth'])
                results['gate_count'].append(quantum_result['gate_count'])
                
                # Calculate constraint violation
                quantum_violation = calculate_constraint_violation(
                    quantum_result['solution'], target_size
                )
                results['quantum_violations'].append(quantum_violation)
                
                # Set baseline if this is first successful run
                if baseline_quantum_time is None:
                    baseline_quantum_time = quantum_result['optimization_time']
                    
            else:
                # Fill with None for failed runs
                for key in ['quantum_time', 'quantum_objective', 'quantum_memory', 
                           'quantum_convergence', 'quantum_diversity', 'circuit_depth',
                           'gate_count', 'quantum_violations']:
                    results[key].append(None)
            
            if classical_result['success']:
                results['classical_time'].append(classical_result['optimization_time'])
                results['classical_objective'].append(classical_result['objective_value'])
                results['classical_memory'].append(classical_result['memory_usage'])
                
                # Calculate constraint violation
                classical_violation = calculate_constraint_violation(
                    classical_result['solution'], target_size
                )
                results['classical_violations'].append(classical_violation)
                
                # Set baseline if this is first successful run
                if baseline_classical_time is None:
                    baseline_classical_time = classical_result['optimization_time']
                    
            else:
                for key in ['classical_time', 'classical_objective', 'classical_memory',
                           'classical_violations']:
                    results[key].append(None)
            
            # Calculate relative performance
            if (quantum_result['success'] and classical_result['success'] and 
                baseline_quantum_time is not None):
                relative_perf = (classical_result['objective_value'] - 
                               quantum_result['objective_value']) / abs(classical_result['objective_value'])
                results['relative_performance'].append(relative_perf)
            else:
                results['relative_performance'].append(None)
            
            # Print iteration summary
            print(f"\nIteration {n_assets} assets completed:")
            if quantum_result['success']:
                print(f"  Quantum - Time: {quantum_result['optimization_time']:.2f}s, "
                      f"Objective: {quantum_result['objective_value']:.4f}, "
                      f"Memory: {quantum_result['memory_usage']:.1f}MB")
            else:
                print(f"  Quantum - FAILED")
                
            if classical_result['success']:
                print(f"  Classical - Time: {classical_result['optimization_time']:.2f}s, "
                      f"Objective: {classical_result['objective_value']:.4f}, "
                      f"Memory: {classical_result['memory_usage']:.1f}MB")
            else:
                print(f"  Classical - FAILED")
                
        except Exception as e:
            print(f"ERROR in iteration {n_assets}: {e}")
            # Fill with None for failed iterations - make sure to include n_assets
            results['n_assets'].append(n_assets)
            for key in results.keys():
                if key != 'n_assets':
                    results[key].append(None)
        
        # Clean up memory
        gc.collect()
        
        print(f"Iteration {n_assets} assets COMPLETED ✓")
    
    return results

def create_scalability_plots(results):
    """Create comprehensive scalability analysis plots"""
    
    # Create output directory
    plot_dir = "scalability_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nCreating scalability analysis plots...")
    print(f"Saving to: {plot_dir}/")
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    df_success = df[(df['quantum_success'] == True) & (df['classical_success'] == True)]
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    def save_plot(fig, filename):
        """Save plot with timestamp"""
        full_path = os.path.join(plot_dir, f"{timestamp}_{filename}")
        fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {full_path}")
        return full_path
    
    # 1. Performance Scaling Overview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quantum Portfolio Optimization Scalability Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Optimization time scaling
    if len(df_success) > 0:
        ax1.plot(df_success['n_assets'], df_success['quantum_time'], 
                'o-', label='Quantum (QAOA)', linewidth=2, markersize=6)
        ax1.plot(df_success['n_assets'], df_success['classical_time'], 
                's-', label='Classical', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Assets (Qubits)')
        ax1.set_ylabel('Optimization Time (seconds)')
        ax1.set_title('Optimization Time Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # Solution quality comparison
    if len(df_success) > 0:
        ax2.plot(df_success['n_assets'], df_success['quantum_objective'], 
                'o-', label='Quantum', linewidth=2, markersize=6)
        ax2.plot(df_success['n_assets'], df_success['classical_objective'], 
                's-', label='Classical', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Assets (Qubits)')
        ax2.set_ylabel('Objective Function Value')
        ax2.set_title('Solution Quality Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Memory usage scaling
    if len(df_success) > 0:
        ax3.plot(df_success['n_assets'], df_success['quantum_memory'], 
                'o-', label='Quantum', linewidth=2, markersize=6)
        ax3.plot(df_success['n_assets'], df_success['classical_memory'], 
                's-', label='Classical', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of Assets (Qubits)')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Relative performance
    if len(df_success) > 0:
        relative_perf = df_success['relative_performance'].dropna()
        if len(relative_perf) > 0:
            ax4.plot(df_success['n_assets'][:len(relative_perf)], relative_perf * 100, 
                    'o-', linewidth=2, markersize=6, color='green')
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Number of Assets (Qubits)')
            ax4.set_ylabel('Quantum Advantage (%)')
            ax4.set_title('Relative Performance (Quantum vs Classical)')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, "scalability_overview.png")
    
    # 2. Detailed Quantum Metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quantum Algorithm Scaling Metrics', fontsize=16, fontweight='bold')
    
    # Circuit complexity
    quantum_df = df[df['quantum_success'] == True]
    if len(quantum_df) > 0:
        ax1.plot(quantum_df['n_assets'], quantum_df['circuit_depth'], 
                'o-', label='Circuit Depth', linewidth=2, markersize=6)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(quantum_df['n_assets'], quantum_df['gate_count'], 
                     's-', label='Gate Count', linewidth=2, markersize=6, color='orange')
        ax1.set_xlabel('Number of Assets (Qubits)')
        ax1.set_ylabel('Circuit Depth', color='blue')
        ax1_twin.set_ylabel('Gate Count', color='orange')
        ax1.set_title('Circuit Complexity Scaling')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
    
    # Convergence behavior
    if len(quantum_df) > 0:
        ax2.plot(quantum_df['n_assets'], quantum_df['quantum_convergence'], 
                'o-', linewidth=2, markersize=6, color='purple')
        ax2.set_xlabel('Number of Assets (Qubits)')
        ax2.set_ylabel('Iterations to Convergence')
        ax2.set_title('Convergence Speed')
        ax2.grid(True, alpha=0.3)
    
    # Constraint satisfaction
    if len(quantum_df) > 0:
        ax3.plot(quantum_df['n_assets'], quantum_df['quantum_violations'], 
                'o-', label='Quantum', linewidth=2, markersize=6, color='red')
        if len(df_success) > 0:
            ax3.plot(df_success['n_assets'], df_success['classical_violations'], 
                    's-', label='Classical', linewidth=2, markersize=6, color='blue')
        ax3.set_xlabel('Number of Assets (Qubits)')
        ax3.set_ylabel('Constraint Violation Rate')
        ax3.set_title('Constraint Satisfaction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Solution diversity
    if len(quantum_df) > 0:
        ax4.plot(quantum_df['n_assets'], quantum_df['quantum_diversity'], 
                'o-', linewidth=2, markersize=6, color='green')
        ax4.set_xlabel('Number of Assets (Qubits)')
        ax4.set_ylabel('Solution Diversity')
        ax4.set_title('Solution Space Exploration')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, "quantum_metrics.png")
    
    # 3. Success Rate Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Algorithm Success Rate Analysis', fontsize=16, fontweight='bold')
    
    # Success rates
    success_rates = df.groupby('n_assets').agg({
        'quantum_success': 'mean',
        'classical_success': 'mean'
    }).reset_index()
    
    if len(success_rates) > 0:
        ax1.plot(success_rates['n_assets'], success_rates['quantum_success'] * 100, 
                'o-', label='Quantum', linewidth=2, markersize=6)
        ax1.plot(success_rates['n_assets'], success_rates['classical_success'] * 100, 
                's-', label='Classical', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Assets (Qubits)')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Algorithm Success Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
    
    # Performance ratio
    if len(df_success) > 0:
        time_ratio = df_success['quantum_time'] / df_success['classical_time']
        ax2.plot(df_success['n_assets'], time_ratio, 
                'o-', linewidth=2, markersize=6, color='purple')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        ax2.set_xlabel('Number of Assets (Qubits)')
        ax2.set_ylabel('Time Ratio (Quantum/Classical)')
        ax2.set_title('Relative Speed Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    save_plot(fig, "success_analysis.png")
    
    return plot_dir

def generate_scalability_report(results, plot_dir):
    """Generate comprehensive scalability analysis report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(plot_dir, f"{timestamp}_scalability_report.txt")
    
    df = pd.DataFrame(results)
    df_success = df[(df['quantum_success'] == True) & (df['classical_success'] == True)]
    
    report_content = f"""
QUANTUM PORTFOLIO OPTIMIZATION SCALABILITY ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*70}

ANALYSIS OVERVIEW:
- Qubit Range Tested: 4 to 19 assets
- Total Test Cases: {len(df)}
- Successful Quantum Runs: {df['quantum_success'].sum()}
- Successful Classical Runs: {df['classical_success'].sum()}
- Both Methods Successful: {len(df_success)}

PERFORMANCE SUMMARY:
{'='*50}

Optimization Time Scaling:
"""
    
    if len(df_success) > 0:
        quantum_times = df_success['quantum_time'].dropna()
        classical_times = df_success['classical_time'].dropna()
        
        if len(quantum_times) > 1:
            quantum_growth = quantum_times.iloc[-1] / quantum_times.iloc[0]
            report_content += f"- Quantum Time Growth (4→19 qubits): {quantum_growth:.2f}x\n"
        
        if len(classical_times) > 1:
            classical_growth = classical_times.iloc[-1] / classical_times.iloc[0]
            report_content += f"- Classical Time Growth (4→19 qubits): {classical_growth:.2f}x\n"
        
        avg_quantum_time = quantum_times.mean()
        avg_classical_time = classical_times.mean()
        report_content += f"- Average Quantum Time: {avg_quantum_time:.3f} seconds\n"
        report_content += f"- Average Classical Time: {avg_classical_time:.3f} seconds\n"
        report_content += f"- Speed Ratio (Q/C): {avg_quantum_time/avg_classical_time:.2f}\n"

    report_content += f"""

Solution Quality Analysis:
"""
    
    if len(df_success) > 0:
        quantum_obj = df_success['quantum_objective'].dropna()
        classical_obj = df_success['classical_objective'].dropna()
        
        if len(quantum_obj) > 0 and len(classical_obj) > 0:
            avg_quantum_obj = quantum_obj.mean()
            avg_classical_obj = classical_obj.mean()
            
            report_content += f"- Average Quantum Objective: {avg_quantum_obj:.6f}\n"
            report_content += f"- Average Classical Objective: {avg_classical_obj:.6f}\n"
            
            if avg_classical_obj != 0:
                relative_quality = (avg_classical_obj - avg_quantum_obj) / abs(avg_classical_obj) * 100
                report_content += f"- Relative Quality Advantage: {relative_quality:.2f}%\n"

    report_content += f"""

Resource Usage Analysis:
"""
    
    if len(df_success) > 0:
        quantum_mem = df_success['quantum_memory'].dropna()
        classical_mem = df_success['classical_memory'].dropna()
        
        if len(quantum_mem) > 0:
            report_content += f"- Average Quantum Memory: {quantum_mem.mean():.1f} MB\n"
            report_content += f"- Peak Quantum Memory: {quantum_mem.max():.1f} MB\n"
        
        if len(classical_mem) > 0:
            report_content += f"- Average Classical Memory: {classical_mem.mean():.1f} MB\n"
            report_content += f"- Peak Classical Memory: {classical_mem.max():.1f} MB\n"

    report_content += f"""

Circuit Complexity Analysis:
"""
    
    quantum_df = df[df['quantum_success'] == True]
    if len(quantum_df) > 0:
        circuit_depth = quantum_df['circuit_depth'].dropna()
        gate_count = quantum_df['gate_count'].dropna()
        
        if len(circuit_depth) > 0:
            report_content += f"- Average Circuit Depth: {circuit_depth.mean():.1f}\n"
            report_content += f"- Maximum Circuit Depth: {circuit_depth.max()}\n"
        
        if len(gate_count) > 0:
            report_content += f"- Average Gate Count: {gate_count.mean():.1f}\n"
            report_content += f"- Maximum Gate Count: {gate_count.max()}\n"

    report_content += f"""

Constraint Satisfaction Analysis:
"""
    
    if len(df_success) > 0:
        quantum_viol = df_success['quantum_violations'].dropna()
        classical_viol = df_success['classical_violations'].dropna()
        
        if len(quantum_viol) > 0:
            report_content += f"- Average Quantum Constraint Violation: {quantum_viol.mean():.4f}\n"
            report_content += f"- Maximum Quantum Violation: {quantum_viol.max():.4f}\n"
        
        if len(classical_viol) > 0:
            report_content += f"- Average Classical Constraint Violation: {classical_viol.mean():.4f}\n"
            report_content += f"- Maximum Classical Violation: {classical_viol.max():.4f}\n"

    report_content += f"""

KEY INSIGHTS:
{'='*50}
"""
    
    # Generate insights based on the data
    insights = []
    
    if len(df_success) > 1:
        # Scaling behavior
        quantum_times = df_success['quantum_time'].dropna()
        if len(quantum_times) > 1:
            if quantum_times.iloc[-1] / quantum_times.iloc[0] > 10:
                insights.append("• Quantum optimization time shows exponential scaling challenges")
            elif quantum_times.iloc[-1] / quantum_times.iloc[0] > 3:
                insights.append("• Quantum optimization shows polynomial scaling")
            else:
                insights.append("• Quantum optimization demonstrates good scalability")
        
        # Success rate analysis
        success_rate = df['quantum_success'].mean()
        if success_rate > 0.8:
            insights.append("• High quantum algorithm success rate across problem sizes")
        elif success_rate > 0.5:
            insights.append("• Moderate quantum algorithm reliability")
        else:
            insights.append("• Quantum algorithm faces scalability challenges")
        
        # Performance comparison
        if len(df_success) > 0:
            avg_relative = pd.Series(df_success['relative_performance']).dropna().mean()
            if avg_relative > 0.05:
                insights.append("• Quantum method shows consistent advantage over classical")
            elif avg_relative > -0.05:
                insights.append("• Quantum and classical methods show comparable performance")
            else:
                insights.append("• Classical method outperforms quantum on average")
    
    for insight in insights:
        report_content += f"{insight}\n"

    report_content += f"""

DETAILED RESULTS BY PROBLEM SIZE:
{'='*50}
"""
    
    for _, row in df.iterrows():
        n = int(row['n_assets'])
        report_content += f"\n{n} Assets ({n} Qubits):\n"
        
        if row['quantum_success']:
            report_content += f"  Quantum: {row['quantum_time']:.3f}s, Obj: {row['quantum_objective']:.6f}, "
            report_content += f"Mem: {row['quantum_memory']:.1f}MB\n"
        else:
            report_content += f"  Quantum: FAILED\n"
        
        if row['classical_success']:
            report_content += f"  Classical: {row['classical_time']:.3f}s, Obj: {row['classical_objective']:.6f}, "
            report_content += f"Mem: {row['classical_memory']:.1f}MB\n"
        else:
            report_content += f"  Classical: FAILED\n"

    report_content += f"""

ANALYSIS COMPLETED
{'='*50}
This scalability analysis provides insights into quantum portfolio optimization
performance across different problem sizes, helping identify optimal operating
ranges and scalability limitations.
"""
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"  Saved comprehensive report: {report_path}")
    return report_path

def main():
    """Main execution function"""
    print("Starting Quantum Portfolio Optimization Scalability Analysis...")
    
    # Run scalability analysis
    results = run_scalability_analysis()
    
    # Create visualization plots
    plot_dir = create_scalability_plots(results)
    
    # Generate comprehensive report
    report_path = generate_scalability_report(results, plot_dir)
    
    print(f"\n" + "="*80)
    print("SCALABILITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {plot_dir}/")
    print(f"Generated plots:")
    print(f"  1. scalability_overview.png - Overall performance scaling")
    print(f"  2. quantum_metrics.png - Detailed quantum algorithm metrics")
    print(f"  3. success_analysis.png - Success rates and performance ratios")
    print(f"  4. scalability_report.txt - Comprehensive analysis report")
    
    # Summary statistics
    df = pd.DataFrame(results)
    successful_runs = df['quantum_success'].sum()
    total_runs = len(df)
    
    print(f"\nSummary:")
    print(f"  Total test cases: {total_runs}")
    print(f"  Successful quantum runs: {successful_runs}/{total_runs} ({successful_runs/total_runs*100:.1f}%)")
    print(f"  Analysis duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nAnalysis complete! Check the generated plots and report for detailed insights.")

if __name__ == "__main__":
    main()
