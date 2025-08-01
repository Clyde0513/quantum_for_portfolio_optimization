# Quantum Portfolio Optimization (**for explore step1-5.py**. It's different for the original step1-5.py because we're just experimenting)

This project demonstrates a quantum approach to portfolio optimization using Variational Quantum Eigensolver (VQE) with QAOA ansatz and PennyLane. The implementation achieves competitive performance with classical methods while demonstrating  constraint satisfaction capabilities.

## Overview

The goal of this project is to solve a Quadratic Unconstrained Binary Optimization (QUBO) problem for portfolio optimization. The problem is mapped to a quantum Hamiltonian, and the ground state of the Hamiltonian represents the optimal portfolio.

### Key Results

- **Problem Size:** 12 bonds, target basket size: 6 (3 in step1-5.py)
- **Quantum Solution:** [1,1,0,0,0,1,0,0,1,0,1,1] with cost -84,412.33 (perfect constraint satisfaction)
- **Classical Benchmark:** [0,1,0,0,0,0,0,1,1,1,1,1] with cost -84,444.28 (perfect constraint satisfaction)
- **Performance Result:** NEAR TIE - Difference: 0.04% (within 1% tolerance)

### Constraint Analysis

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation |
|------------------|-------------|-----------|----------------|-----------------|
| **Quantum**      | 6           | 0.0300    | 0.8662         | **0.0000**      |
| **Classical**    | 6           | 0.0364    | 0.8328         | **0.0000**      |

### Enhanced VQE Implementation

- **Circuit Architecture:** QAOA ansatz with 4 layers for optimal expressivity
- **Multiple Restarts:** 5 restarts × 500 iterations each for robust optimization
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
   python "exploring step1-5.py"
   ```

3. Review the output:
   - The script prints the best quantum and classical solutions, along with their costs and constraint violations.
   - Top 5 quantum solutions with frequency and feasibility analysis
   - Comprehensive constraint satisfaction metrics

## Example Output for exploring steps1-5.py

```text
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

Starting VQE optimization with QAOA and adaptive penalties...

--- Restart 1/5 ---
C:\Users\iclyd\anaconda3\envs\vqe\Lib\site-packages\pennylane\ops\op_math\composite.py:211: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in staticmethod() if you want to preserve the old behavior
  return self._math_op(math.vstack(eigvals), axis=0)
 Iteration   0, Cost: -111439.4875
 Iteration 100, Cost: -111608.3884
 Iteration 200, Cost: -111609.6921
 Iteration 300, Cost: -111528.0366
 Iteration 400, Cost: -112360.9864
 Final cost for restart 1: -111767.8944

--- Restart 2/5 ---
 Iteration   0, Cost: -112055.0328
 Iteration 100, Cost: -112564.1518
 Iteration 200, Cost: -111775.3272
 Iteration 300, Cost: -112150.8303
 Iteration 400, Cost: -111260.4701
 Final cost for restart 2: -111427.8796

--- Restart 3/5 ---
 Iteration   0, Cost: -112185.3799
 Iteration 100, Cost: -111814.5979
 Iteration 200, Cost: -111656.5295
 Iteration 300, Cost: -111951.4403
 Iteration 400, Cost: -112469.0214
 Final cost for restart 3: -111942.8882

--- Restart 4/5 ---
 Iteration   0, Cost: -111326.1261
 Iteration 100, Cost: -111452.6269
 Iteration 200, Cost: -112602.9614
 Iteration 300, Cost: -111643.4030
 Iteration 400, Cost: -111577.7264
 Final cost for restart 4: -111282.8246

--- Restart 5/5 ---
 Iteration   0, Cost: -111834.0263
 Iteration 100, Cost: -111828.0410
 Iteration 200, Cost: -111896.9374
 Iteration 300, Cost: -112836.8377
 Iteration 400, Cost: -111981.6313
 Final cost for restart 5: -111672.9459

Best VQE cost: -111942.8882

Sampling from optimized circuit...
 Feasible quantum solution: [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0] (cost: -72337.61, freq: 140)
 Feasible quantum solution: [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0] (cost: -84202.21, freq: 80)
 Feasible quantum solution: [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1] (cost: -84340.17, freq: 69)
 Feasible quantum solution: [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1] (cost: -84412.33, freq: 65)

Best quantum solution: [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
Quantum cost: -84412.3263

Solving classically with simulated annealing...
Best classical solution: [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Classical cost: -84444.2759

Quantum Solution Analysis:
 Portfolio: [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
 Basket size: 6 (target: 6) - Violation: 0
 Cash flow: 0.0300 (range: [0.0100, 0.0500]) - Violation: 0.0000
 Characteristic: 0.8662 (range: [0.6000, 1.2000]) - Violation: 0.0000
 Tracking error: 3.9721
 Total constraint violation: 0.0000

Classical Solution Analysis:
 Portfolio: [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 Basket size: 6 (target: 6) - Violation: 0
 Cash flow: 0.0364 (range: [0.0100, 0.0500]) - Violation: 0.0000
 Characteristic: 0.8328 (range: [0.6000, 1.2000]) - Violation: 0.0000
 Tracking error: 2.9958
 Total constraint violation: 0.0000

=== Final Summary ===
Problem size: 12 bonds, Target basket: 6
QAOA layers: 4, Restarts: 5, Iterations per restart: 500
Best quantum cost: -84412.3263 (violation: 0.0000)
Best classical cost: -84444.2759 (violation: 0.0000)
Cost ratio (quantum/classical): 1.000
NEAR TIE - Difference: 0.04% (within 1% tolerance)

Top 5 quantum solutions:
 1. [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0] (freq:  140, cost: -72337.61, feasible: False)
 2. [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0] (freq:   80, cost: -84202.21, feasible: False)
 3. [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0] (freq:   79, cost: -44293.51, feasible: False)
 4. [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1] (freq:   79, cost: -84113.14, feasible: False)
 5. [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1] (freq:   76, cost: -84109.82, feasible: True)
 ```

## Example Output for steps1-5.py
```
 === Quantum Portfolio Optimization ===

Bond amounts if selected (x_c): [2.54187582 3.00880099 1.7043105  3.56029985 2.21471029 2.21633431]
Target basket size: 3
Cash flow range: [0.0100, 0.0500]
Characteristic range: [0.6000, 1.0000]

Building QUBO formulation...
Cash flow coefficients (a_cf): [0.00476531 0.00635328 0.00508813 0.00443377 0.00277605 0.00315832]
QUBO matrix Q shape: (6, 6)
QUBO vector q shape: (6,)

Hamiltonian constructed with 58 terms

Starting VQE optimization with multiple restarts...

--- Restart 1/3 ---
C:\Users\iclyd\anaconda3\envs\vqe\Lib\site-packages\pennylane\ops\op_math\composite.py:211: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in staticmethod() if you want to preserve the old behavior
  return self._math_op(math.vstack(eigvals), axis=0)
  Iteration   0, Cost: -28890.1530
  Iteration  50, Cost: -41160.7068
  Iteration 100, Cost: -42227.3638
  Iteration 150, Cost: -42406.8845
  Iteration 200, Cost: -42561.4872
  Final cost for restart 1: -42588.2481
  *** New best cost: -42588.2481 ***

--- Restart 2/3 ---
  Iteration   0, Cost: -30055.5657
  Iteration  50, Cost: -41389.4713
  Iteration 100, Cost: -42207.4783
  Iteration 150, Cost: -42422.6896
  Iteration 200, Cost: -42482.1647
  Final cost for restart 2: -42501.5222

--- Restart 3/3 ---
  Iteration   0, Cost: -28973.2680
  Iteration  50, Cost: -41757.2449
  Iteration 100, Cost: -42474.0939
  Iteration 150, Cost: -42538.7883
  Iteration 200, Cost: -42559.1654
  Final cost for restart 3: -42580.7646

Best VQE cost across all restarts: -42588.2481

Sampling from optimized circuit...

Most frequent quantum solution: [1, 1, 1, 1, 1, 1]
Frequency: 19792/20000 (99.0%)

Analyzing top quantum solutions for constraint satisfaction...
  Found feasible quantum solution: [0, 1, 0, 1, 1, 0] (cost: -24273.38, freq: 4)
  Found feasible quantum solution: [1, 1, 0, 0, 1, 0] (cost: -24316.15, freq: 2)
  Found feasible quantum solution: [1, 0, 0, 1, 1, 0] (cost: -24329.65, freq: 1)

Using best feasible quantum solution instead of most frequent:
Best feasible quantum solution: [1, 0, 0, 1, 1, 0]
Found 7 total feasible quantum solutions

Solving classically for benchmark...
Best classical solution: (tensor(1, requires_grad=True), tensor(0, requires_grad=True), tensor(1, requires_grad=True), tensor(0, requires_grad=True), tensor(1, requires_grad=True), tensor(1, requires_grad=True))
Classical cost: -24354.2873
Quantum solution cost: -24329.6473

=== Detailed Solution Analysis ===

Classical Solution Analysis:
  Portfolio: [1, 0, 1, 0, 1, 1]
  Basket size: 4 (target: 3) - Violation: 1
  Cash flow: 0.0158 (range: [0.0100, 0.0500]) - Violation: 0.0000
  Characteristic: 0.6313 (range: [0.6000, 1.0000]) - Violation: 0.0000
  Tracking error: 1.9146
  Total constraint violation: 1.0000

Quantum Solution Analysis:
  Portfolio: [1, 0, 0, 1, 1, 0]
  Basket size: 3 (target: 3) - Violation: 0
  Cash flow: 0.0120 (range: [0.0100, 0.0500]) - Violation: 0.0000
  Characteristic: 0.7051 (range: [0.6000, 1.0000]) - Violation: 0.0000
  Tracking error: 3.3021
  Total constraint violation: 0.0000

=== Analyzing Feasible Solutions (exactly 3 bonds) ===
Best feasible solution: [1, 0, 1, 0, 1, 0]
Best feasible cost: -24350.9200

Best Feasible Solution Analysis:
  Portfolio: [1, 0, 1, 0, 1, 0]
  Basket size: 3 (target: 3) - Violation: 0
  Cash flow: 0.0126 (range: [0.0100, 0.0500]) - Violation: 0.0000
  Characteristic: 0.6073 (range: [0.6000, 1.0000]) - Violation: 0.0000
  Tracking error: 1.8116
  Total constraint violation: 0.0000

=== Final Summary ===
Problem size: 6 bonds, Target basket: 3
VQE layers: 8, Restarts: 3, Iterations per restart: 250
Best quantum cost: -24329.6473 (violation: 0.0000)
Best classical cost: -24354.2873 (violation: 1.0000)
Cost ratio (quantum/classical): 0.999
QUANTUM WINS: Better constraint satisfaction

QUANTUM ADVANTAGE ACHIEVED through constraint satisfaction
   Quantum: [1, 0, 0, 1, 1, 0] (violations: 0.0000)
   Classical: [1, 0, 1, 0, 1, 1] (violations: 1.0000)

Top 5 quantum solutions:
  1. [1, 1, 1, 1, 1, 1] (freq: 19792, cost: -12166.53, feasible: False)
  2. [1, 1, 0, 1, 1, 1] (freq:  112, cost: -20202.74, feasible: False)
  3. [0, 1, 1, 1, 1, 1] (freq:   32, cost: -20264.35, feasible: False)
  4. [1, 0, 1, 1, 1, 1] (freq:   17, cost: -20319.61, feasible: False)
  5. [0, 1, 1, 1, 1, 0] (freq:    8, cost: -24266.94, feasible: False)

Top 5 classical solutions:
  1. [1, 0, 1, 0, 1, 0] (cost: -24350.92, feasible: True)
  2. [1, 0, 0, 0, 1, 1] (cost: -24345.27, feasible: True)
  3. [1, 1, 1, 0, 0, 0] (cost: -24332.97, feasible: True)
  4. [1, 1, 0, 0, 0, 1] (cost: -24331.21, feasible: True)
  5. [1, 0, 0, 1, 1, 0] (cost: -24329.65, feasible: True)
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
