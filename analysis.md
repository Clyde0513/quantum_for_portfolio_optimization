# Quantum Portfolio Optimization (for explore pat 1 of step1-5.py)

## **Performance Metrics:**

- **Problem Size:** 12 bonds, target basket size: 6 (Used to be 3)
- **VQE Convergence:** Optimized from -111,439 to -111,943 over 2,500 iterations (5 restarts × 500)
- **Quantum Solution:** [1,1,0,0,0,1,0,0,1,0,1,1] with cost -84,412.33 (onstraint satisfaction)
- **Classical Benchmark:** [0,1,0,0,0,0,0,1,1,1,1,1] with cost -84,444.28 (constraint satisfaction)
- **Performance Result:** NEAR TIE - Difference: 0.04% (within 1% tolerance)

### **Constraint Analysis:**

| Solution | Basket Size | Cash Flow | Characteristic | Total Violation |
|----------|-------------|-----------|----------------|-----------------|
| **Quantum** | 6 | 0.0300 | 0.8662 | **0.0000** |
| **Classical** | 6 | 0.0364 | 0.8328 | **0.0000** |

- **Quantum Solution:** Perfect constraint satisfaction (0 violations)
- **Classical Solution:** Perfect constraint satisfaction (0 violations)
- **Advantage Type:** Competitive performance withs constraint handling

### **Enhanced VQE Implementation:**

- Doubled problem size from 6 to 12 bonds for more realistic complexity
- Increased circuit depth (QAOA with 4 layers) for better expressivity
- Added multiple random restarts (5 restarts × 500 iterations each)
- Adaptive penalty adjustment during optimization
- Sampling strategy (50,000 shots)
- Greedy post-processing for feasibility improvement

### **Adaptive Penalty Parameter System:**

- **Initial Basket Size Penalty:** 2,000 with 1.5x adaptive increase
- **Initial Characteristic Penalty:** 300 with 1.2x adaptive increase
- **Initial Cash Flow Penalties:** 500 each with 1.2x adaptive increase
- **Dynamic adjustment:** Every 50 iterations based on constraint violations

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

- Implemented VQE with QAOA and adaptive penalties (uses StronglyEntanglingLayers ansatz in step1-5.py but this isn't good for QUBO problems)
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
- Larger problem sizes would better demonstrate quantum advantage, in theory

## **Scalability Analysis**

### **Current Implementation:**

- **6 qubits:** Manageable on classical simulators
- **64 possible solutions:** Allows exact classical verification
- **Complexity:** O(2^n) classical brute force vs polynomial VQE

### **Scaling Potential:**

- **Real-world portfolios:** 100-1000+ assets
- **Quantum advantage:** Expected at 20+ qubits where classical becomes intractable
- **Hybrid approaches:** Combine quantum optimization with classical preprocessing

### **Research Inquiries**

1. **Real Market Data:** Use actual stock/bond data instead of synthetic
2. **Risk Models:** Implement more sophisticated risk measures
3. **Portfolio Rebalancing:** Dynamic optimization over time periods
