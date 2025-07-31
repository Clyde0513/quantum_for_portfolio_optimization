# Quantum Portfolio Optimization

## **Performance Metrics:**

- **Problem Size:** 6 bonds, target basket size: 3
- **VQE Convergence:** Successfully optimized from -28,890 to -42,588 over 750 iterations (3 restarts × 250)
- **Quantum Solution:** [1,0,0,1,1,0] with cost -24,329.65 (constraint satisfaction)
- **Classical Benchmark:** [1,0,1,0,1,1] with cost -24,354.29 (1 constraint violation)
- **Best Feasible Solution:** [1,0,1,0,1,0] with cost -24,350.92 (exactly 3 bonds)

### **Constraint Analysis:**

| Solution | Basket Size | Cash Flow | Characteristic | Total Violation |
|----------|-------------|-----------|----------------|-----------------|
| **Quantum** | 3 | 0.0120 | 0.7051 | **0.0000** |
| **Classical** | 4 (violation: 1) | 0.0158 | 0.6313 | **1.0000** |
| **Best Feasible** | 3 | 0.0126 | 0.6073 | **0.0000** |

- **Quantum Solution:** constraint satisfaction (0 violations)
- **Classical Solution:** Violates basket size constraint (1 violation)
- **Advantage Type:** Constraint satisfaction with competitive objective value

### **Enhanced VQE Implementation:**

- Increased circuit depth (8 layers) for better expressivity
- Added multiple random restarts (3 restarts × 250 iterations each)
- Improved optimization parameters and penalty weights
- Better sampling strategy (20,000 shots)
- Analyzed top 20 solutions for constraint feasibility

### **Aggressive Penalty Parameter Tuning:**

- **Basket Size Penalty:** Increased from 200 → 2,000 (10x increase)
- **Characteristic Penalty:** Increased from 30 → 300 (10x increase)
- **Cash Flow Penalties:** Increased to 500 each

### **Benchmarking:**

- Classical brute-force solver for exact comparison
- Constraint satisfaction analysis
- Feasibility checking for all possible solutions

**Task 1 (Mathematical Review):**

- Identified binary decision variables, linear constraints, and quadratic objective
- Understanding of the portfolio optimization formulation from the challenge image

**Task 2 (Quantum Formulation):**

- Converted constrained problem to unconstrained using penalty methods
- Proper QUBO formulation with all constraints included

**Task 3 (Quantum Program):**

- Implemented VQE with StronglyEntanglingLayers ansatz
- Hamiltonian construction and optimization

**Task 4 (Solve with Quantum):**

- Optimized and found quantum solutions
- Proper sampling and result interpretation

**Task 5 (Classical Comparison):**

- Implemented classical brute-force benchmark
- Performance comparison with metrics

### **Algorithm Performance:**

**Strengths:**

- VQE shows good convergence behavior
- Quantum algorithm explores different solution regions
- Handles multiple constraints simultaneously
- Proper constraint violation analysis

**Areas for Improvement:**

- Penalty parameter tuning needed for better constraint satisfaction
- Could benefit from hybrid classical-quantum preprocessing
- Larger problem sizes would better demonstrate quantum advantage

## **Scalability Analysis**

### **Current Implementation:**

- **6 qubits:** Manageable on classical simulators
- **64 possible solutions:** Allows exact classical verification
- **Complexity:** O(2^n) classical brute force vs polynomial VQE

### **Scaling Potential:**

- **Real-world portfolios:** 100-1000+ assets
- **Quantum advantage:** Expected at 20+ qubits where classical becomes intractable
- **Hybrid approaches:** Combine quantum optimization with classical preprocessing

### **Improvements (hopefully it'll allow for different bonds)**

1. **Multiple VQE Runs:** Average results over multiple random initializations
2. **Advanced Penalty Tuning:** Systematic approach to find optimal λ values
3. **Smart Solution Selection:** Analyze multiple solutions for feasibility
4. **Noise Modeling:** Add realistic quantum hardware noise models

5. **QAOA Implementation:** Compare VQE with Quantum Approximate Optimization Algorithm
6. **Real Market Data:** Use actual stock/bond data instead of synthetic
7. **Risk Models:** Implement more sophisticated risk measures
8. **Portfolio Rebalancing:** Dynamic optimization over time periods
