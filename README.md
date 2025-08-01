# Quantum Portfolio Optimization 

(**for exploring part 2 of step1-5.py**)

It's different for the exploring part 1 of step1-5.py because we're just experimenting

This project demonstrates a quantum approach to portfolio optimization using Variational Quantum Eigensolver (VQE) with QAOA ansatz and PennyLane. The implementation achieves competitive performance with classical methods while demonstrating  constraint satisfaction capabilities. This is a Hackathon for Wiser's Project 2: Quantum for Portfolio Optimization.

## Overview

The goal of this project is to solve a Quadratic Unconstrained Binary Optimization (QUBO) problem for portfolio optimization. The problem is mapped to a quantum Hamiltonian, and the ground state of the Hamiltonian represents the optimal portfolio.

### Key Results

- **Problem Size:** 16 bonds, target basket size: 8 (6 in step1-5.py)
- **Quantum Solution:** [1,0,0,1,1,1,0,0,1,0,0,0,0,1,1,1] with cost -144418.5019 (constraint satisfaction)
- **Classical Benchmark:** [1,0,1,1,0,0,0,1,1,1,0,0,0,1,0,1] with cost -144257.2160 (constraint satisfaction)
- **Cash Flow and Characteristic Ranges, respectively** (range: [0.0100, 0.0500]) AND (range: [0.6000, 1.2000])
- **Performance Result:** QUANTUM ADVANTAGE: Better constraint satisfaction
- **Note:** Note that the classical's characteristic is 1.3292 which is outside of the characteristic range, however, if we up the range by a bit, then we think that it will have the same result as part 1--, which is BETTER OBJECTIVE VALUE for quantum

### Constraint Analysis

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation |
|------------------|-------------|-----------|----------------|-----------------|
| **Quantum**      | 8           | 0.0284    | 1.1443         | **0.0000**      |
| **Classical**    | 8           | 0.0397    | 1.3292         | **0.1292**      |

### Enhanced VQE Implementation

- **Circuit Architecture:** QAOA ansatz with 4 layers for optimal expressivity; Also utilizing LIGHTINING.QUBIT instead of DEFAULT.QUBIT
- **Multiple Restarts:** 4 restarts × 200 iterations each for robust optimization
- **Adative Learning Rate:** Use adaptive learning rate of 0.05 during optimization
- **Adaptive Penalties:** Dynamic constraint weight adjustment during optimization
- **Sampling Strategy:** 50,000 shots for high-precision measurements
- **Post-Processing:** Greedy feasibility correction for quantum solutions

## Scalability Analysis

- **Current Implementation:**
  - **12 qubits:** Substantial classical simulation requirements
  - **4,096 possible solutions:** Classical enumeration becomes challenging
  - **Complexity:** Exponential classical search space vs polynomial VQE optimization

- **Scaling Potential:**
  - **Real-world portfolios:** 50-500+ assets
  - **Quantum advantage:** Expected at 20+ qubits where classical optimization becomes intractable
  - **Hybrid approaches:** Quantum sampling with classical post-processing proves effective

## How to Run

1. Install dependencies:

   ```bash
   pip install pennylane matplotlib scipy
   ```

2. Run the script:

   ```bash
   python "exploring part 2 of step1-5.py"
   ```

3. Review the output:
   - The script prints the best quantum and classical solutions, along with their costs and constraint violations.
   - Top 10 quantum solutions with frequency and feasibility analysis
   - Comprehensive constraint satisfaction metrics

## Example Output for exploring part 2 of steps1-5.py
```
=== Exploring (part 2) of Quantum Portfolio Optimization... ===


Problem size: 16 bonds, Target basket: 8
Market parameters (m): [0.68727006 0.97535715 0.86599697 0.79932924 0.57800932 0.57799726
 0.52904181 0.93308807 0.80055751 0.85403629 0.51029225 0.98495493
 0.91622132 0.60616956 0.59091248 0.59170225]

Market parameters (M): [1.30424224 1.52475643 1.43194502 1.29122914 1.61185289 1.13949386
 1.29214465 1.36636184 1.45606998 1.78517596 1.19967378 1.51423444
 1.59241457 1.04645041 1.60754485 1.17052412]

Risk characteristics (i_c): [0.23903096 0.76933132 0.77937922 0.68503841 0.38276826 0.25860327
 0.61053982 0.4640915  0.27322294 0.49710615 0.22063311 0.74559224
 0.35526799 0.59751337 0.38702665 0.51204081]

Bond amounts if selected (x_c): [1.45332161 5.01515063 1.68641043 1.80996991 1.00964527 0.91345709
 1.68000597 1.49032585 3.96530366 3.78697969 3.09475694 3.75989904
 2.48852067 2.88598149 1.13319867 2.27387107]

Target basket size: 8

Cash flow range: [0.0100, 0.0500]
Characteristic range: [0.6, 1.2]
Building QUBO formulation...
QUBO matrix Q shape: (16, 16)
Building Optimized Hamiltonian...
Hamiltonian constructed with 153 terms
Using lightning.qubit: Optimized CPU
Main computation device: lightning.qubit

Starting Optimized-Accelerated VQE training...

--- Restart 1/4 ---
C:\Users\iclyd\anaconda3\envs\vqe\Lib\site-packages\pennylane\devices\preprocess.py:289: UserWarning: Differentiating with respect to the input parameters of LinearCombination is not supported with the adjoint differentiation method. Gradients are computed only with regards to the trainable parameters of the circuit.

 Mark the parameters of the measured observables as non-trainable to silence this warning.
  warnings.warn(
 Iteration   0, Cost: -196459.6843
 Iteration  50, Cost: -196843.8390
 Iteration 100, Cost: -196841.2709
 Iteration 150, Cost: -196412.4516
 Final cost: -196264.1269

--- Restart 2/4 ---
 Iteration   0, Cost: -196694.6439
 Iteration  50, Cost: -196693.5695
 Iteration 100, Cost: -197179.9522
 Iteration 150, Cost: -196965.8573
 Final cost: -196979.3258

--- Restart 3/4 ---
 Iteration   0, Cost: -197698.7425
 Iteration  50, Cost: -196749.8860
 Iteration 100, Cost: -196608.8522
 Iteration 150, Cost: -196899.2819
 Final cost: -196059.2591

--- Restart 4/4 ---
 Iteration   0, Cost: -197833.4262
 Iteration  50, Cost: -196910.9529
 Iteration 100, Cost: -196535.1406
 Iteration 150, Cost: -196479.2931
 Final cost: -196630.7056

Training completed in 278.37 seconds
Best VQE cost: -196979.3258

Optimized-accelerated sampling...
Using lightning.qubit: Optimized CPU
Sampling device: lightning.qubit
Sampling completed in 0.13 seconds

Top quantum solutions (post-processed):
  1. [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0] (cost: -144350.00, freq:   11)
  2. [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0] (cost: -144209.65, freq:   11)
  3. [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1] (cost: -144418.50, freq:   11)
  4. [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0] (cost: -143500.42, freq:   11)
  5. [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1] (cost: -144063.00, freq:   11)
  6. [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0] (cost: -144367.40, freq:   10)
  7. [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1] (cost: -143896.25, freq:   10)
  8. [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0] (cost: -144106.73, freq:   10)
  9. [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0] (cost: -144280.73, freq:   10)
 10. [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0] (cost: -143646.15, freq:   10)

Classical benchmark...

=== Optimized Performance Summary (with lightning) ===
Hardware: lightning.qubit
Problem size: 16 bonds, Target basket: 8
QAOA layers: 4, Restarts: 4

Timing:
  Training: 278.37s
  Sampling: 0.13s
  Classical: 0.08s
  Total quantum: 278.50s

Results:
  Best quantum cost: -144418.5019
  Best classical cost: -144257.2160
  Quantum advantage: 0.999x

Quantum Solution:
  Portfolio: [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]
  Basket size: 8 (target: 8)
  Cash flow: 0.0284 (range: [0.0100, 0.0500])
  Characteristic: 1.1443 (range: [0.6000, 1.2000])
  Constraint violation: 0.0000

Classical Solution:
  Portfolio: [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1]
  Basket size: 8 (target: 8)
  Cash flow: 0.0397 (range: [0.0100, 0.0500])
  Characteristic: 1.3292 (range: [0.6000, 1.2000])
  Constraint violation: 0.1292

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction
```

## Example Output for exploring part 1 of steps1-5.py

```text
== Exploring (part 1) Quantum Portfolio Optimization ===

Bond amounts if selected (x_c): [2.61648762 5.91959588 1.58659725 2.36880602 3.80827142 1.35180569
 4.80666692 1.33203313 2.55412089 2.22373694 2.8699618  2.71078867]
Target basket size: 6
Cash flow range: [0.0100, 0.0500]
Characteristic range: [0.6, 1.2]

Building QUBO formulation...
Cash flow coefficients (a_cf): [0.0039893  0.00802945 0.0051345  0.00522709 0.00327575 0.00232895
 0.00289272 0.0057637  0.00416125 0.00693207 0.00329055 0.00822434]
QUBO matrix Q shape: (12, 12)

Hamiltonian constructed with 223 terms
Using lightning.qubit device for CPU optimization

Starting VQE optimization with QAOA and adaptive penalties...

--- Restart 1/5 ---
C:\Users\iclyd\anaconda3\envs\vqe\Lib\site-packages\pennylane\devices\preprocess.py:289: UserWarning: Differentiating with respect to the input parameters of LinearCombination is not supported with the adjoint differentiation method. Gradients are computed only with regards to the trainable parameters of the circuit.

 Mark the parameters of the measured observables as non-trainable to silence this warning.
  warnings.warn(
 Iteration   0, Cost: -111439.4875
 Iteration 100, Cost: -111133.4326
 Iteration 200, Cost: -111334.4446
 Iteration 300, Cost: -111727.1178
 Iteration 400, Cost: -111675.1916
 Final cost for restart 1: -111621.6214

--- Restart 2/5 ---
 Iteration   0, Cost: -112055.0328
 Iteration 100, Cost: -111529.4948
 Iteration 200, Cost: -111985.7184
 Iteration 300, Cost: -111718.2236
 Iteration 400, Cost: -111585.0230
 Final cost for restart 2: -112098.4695

--- Restart 3/5 ---
 Iteration   0, Cost: -112185.3799
 Iteration 100, Cost: -112145.4474
 Iteration 200, Cost: -112185.1417
 Iteration 300, Cost: -111665.1162
 Iteration 400, Cost: -111888.0429
 Final cost for restart 3: -111183.5832

--- Restart 4/5 ---
 Iteration   0, Cost: -111326.1261
 Iteration 100, Cost: -111719.5029
 Iteration 200, Cost: -111509.4665
 Iteration 300, Cost: -112206.6841
 Iteration 400, Cost: -111891.4744
 Final cost for restart 4: -111409.6408

--- Restart 5/5 ---
 Iteration   0, Cost: -111834.0263
 Iteration 100, Cost: -111521.5622
 Iteration 200, Cost: -111448.4181
 Iteration 300, Cost: -110654.0453
 Iteration 400, Cost: -111976.8439
 Final cost for restart 5: -111994.3537

Best VQE cost: -112098.4695

Sampling from optimized circuit...
Using lightning.qubit for sampling
 Feasible quantum solution: [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0] (cost: -84373.64, freq: 151)

Best quantum solution: [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
Quantum cost: -84373.6355

Solving classically with simulated annealing...
Best classical solution: [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
Classical cost: -84365.6673

Quantum Solution Analysis:
 Portfolio: [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
 Basket size: 6 (target: 6) - Violation: 0
 Cash flow: 0.0312 (range: [0.0100, 0.0500]) - Violation: 0.0000
 Characteristic: 1.0375 (range: [0.6000, 1.2000]) - Violation: 0.0000
 Tracking error: 4.6242
 Total constraint violation: 0.0000

Classical Solution Analysis:
 Portfolio: [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
 Basket size: 6 (target: 6) - Violation: 0
 Cash flow: 0.0356 (range: [0.0100, 0.0500]) - Violation: 0.0000
 Characteristic: 1.1020 (range: [0.6000, 1.2000]) - Violation: 0.0000
 Tracking error: 4.5106
 Total constraint violation: 0.0000

=== Final Summary ===
Problem size: 12 bonds, Target basket: 6
QAOA layers: 4, Restarts: 5, Iterations per restart: 500
Best quantum cost: -84373.6355 (violation: 0.0000)
Best classical cost: -84365.6673 (violation: 0.0000)
Cost ratio (quantum/classical): 1.000
QUANTUM ADVANTAGE ACHIEVED (better objective value)

Top 5 quantum solutions:
 1. [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0] (freq:  151, cost: -84373.64, feasible: True)
 2. [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0] (freq:  101, cost: -80311.91, feasible: False)
 3. [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0] (freq:   99, cost: -72281.93, feasible: False)
 4. [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1] (freq:   87, cost: -72431.63, feasible: False)
 5. [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0] (freq:   84, cost: -83925.36, feasible: True)

Top 5 Quantum Solutions Data for Plotting:
Solution 1: Cost = -84373.64, Frequency = 151
Solution 2: Cost = -80311.91, Frequency = 101
Solution 3: Cost = -72281.93, Frequency = 99
Solution 4: Cost = -72431.63, Frequency = 87
Solution 5: Cost = -83925.36, Frequency = 84
 ```

## Key Features

### Adaptive Penalty System

- Dynamic constraint weight adjustment during optimization
- Automatic penalty scaling based on violation severity
- Improved convergence to feasible solutions

### QAOA Implementation

- Quantum Approximate Optimization Algorithm with 4 layers
- Optimized parameter initialization and learning rates
- Robust multi-restart strategy for global optimization

### Post-Processing Pipeline

- Greedy feasibility correction for quantum solutions
- Statistical analysis of solution frequency and quality
- Constraint violation tracking

### Performance Analysis

- Detailed constraint satisfaction metrics
- Classical benchmark comparison with simulated annealing
- Cost ratio analysis and competitive assessment

## Technical Implementation

- **Quantum Framework:** PennyLane with default.qubit simulator
- **Optimization:** Adam optimizer with adaptive learning rates
- **Sampling:** 50,000 shots for high-precision measurements
- **Classical Benchmark:** Simulated annealing with basin hopping
- **Constraint Handling:** Penalty method with adaptive scaling

## References

- [The Wiser's Quantum Portfolio Optimization](https://www.thewiser.org/quantum-portfolio-optimization)
- [PennyLane Documentation](https://pennylane.ai/)
