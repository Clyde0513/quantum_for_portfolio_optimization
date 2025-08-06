# Quantum Portfolio Optimization

(**for exploring part 3 of step1-5.py**)

It's different for the exploring part 1 [and2] of step1-5.py because we're just experimenting

This project demonstrates a quantum approach to portfolio optimization using Variational Quantum Eigensolver (VQE) with QAOA ansatz and PennyLane. The implementation achieves competitive performance with classical methods while demonstrating  constraint satisfaction capabilities. This is a Hackathon for Wiser's Project 2: Quantum for Portfolio Optimization.

## Overview

The goal of this project is to solve a Quadratic Unconstrained Binary Optimization (QUBO) problem for portfolio optimization. The problem is mapped to a quantum Hamiltonian, and the ground state of the Hamiltonian represents the optimal portfolio.

### Key Results (for the first 5 results, it's all synthetic (realistic))

- **Problem Size:** 16 bonds, target basket size: 8 (6 in step1-5.py)
- **Quantum Solution:** [1,0,0,1,1,1,0,0,1,0,0,0,0,1,1,1] with cost -144418.5019 (constraint satisfaction)
- **Classical Benchmark:** [1,0,1,1,0,0,0,1,1,1,0,0,0,1,0,1] with cost -144257.2160 (constraint satisfaction)
- **Cash Flow and Characteristic Ranges, respectively** (range: [0.0100, 0.0500]) AND (range: [0.6000, 1.2000])
- **Performance Result:** QUANTUM ADVANTAGE: Better constraint satisfaction
- **Note:** Note that the classical's characteristic is 1.3292 which is outside of the characteristic range, however, if we up the range by a bit, then we think that it will have the same result as part 1--, which is BETTER OBJECTIVE VALUE for quantum

### Constraint Analysis of 16 bonds, basket size 8

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost |
|------------------|-------------|-----------|----------------|-----------------|------|
| **Quantum**      | 8           | 0.0284    | 1.1443         | **0.0000**      |-144418|
| **Classical**    | 8           | 0.0397    | 1.3292         | **0.1292**      |-144257|

### Constraint Analysis of 20 bonds, basket size 10 (this is because we didn't increase the ranges for cash flow and characteristic but in theory and in practice, there would not be any constraint violations in either one of them but quantum will still win due to objective value)

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost |
|------------------|-------------|-----------|----------------|-----------------|------|
| **Quantum**      | 10          | 0.0405    | 1.3335         | **0.1335**      |-220171|
| **Classical**    | 10          | 0.0487    | 1.9542         | **1.7542**      |-219163|

### Constraint Analysis of 20 bonds, basket size 10 (This time we increased the ranges for cash flow and characteristic by a bit, so as you can see, quantum still beats classical by violation!)

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost |
|------------------|-------------|-----------|----------------|-----------------|------|
| **Quantum**      | 10          | 0.0417    | 1.3150         | **0.0000**      |-220570|
| **Classical**    | 10          | 0.0464    | 1.7846         | **0.1846**      |-220112|

## WITH Lightning.GPU

### Constraint Analysis of 20 bonds, basket size 10 (same ranges as the one right above this)

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      | 10          | 0.0459    | 1.3502         | **0.0000**      |-220550 |
| **Classical**    | 10          | 0.0416    | 1.6062         | **0.0062**      |-220360 |

### Constraint Analysis of 20 bonds, basket size 10 (this time no constraints because we made the characteristic ranges from 1.6 to 2.0 but quantum still wins) [There's graphs for this in the folder called quantum_analysis]

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      | 10          | 0.0416    | 1.7418         | **0.0000**      |-221781 |
| **Classical**    | 10          | 0.0416    | 1.6521         | **0.0000**      |-221776 |

- **Lightning.qubit Vs Lightning.gpu Speedup:** Lightning.gpu with 20 qubits takes roughly 1700 seconds than Lightning.qubit with 20 qubits which took roughly 8300 seconds. So fast!
- **Using Significant Gates & recommended hardware configs:** So that we speedup the process even more and made sure to recommend optimal configuration based on hardware

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Real Vanguard Data Example 1

### Constraint Analysis of 20 bonds, basket size 10 (Some constraints because we made the characteristic ranges from 1.6 to 2.0 but quantum still wins) [There's graphs for this in the folder called vanguard_quantum_analysis]

## In the next run, we will make the characteristic range realistic now, just experimenting with it

Basket Size: min(8, max(5, n // 3)), where n is the actual number of bonds loaded
-- So if we have 20 bonds --> min (8, max(5, 6)) --> min(8,6) --> 6 basket size

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      |      6      | 0.0663    | 0.2698         | **1.3365**      |-42359  |
| **Classical**    |      6      | 0.0766    | 0.2633         | **2.3533**      |-42351  |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Real Vanguard Data Example 2

### Constraint Analysis of 20 bonds, basket size 10 (Characteristic ranges from 0.4 to 0.8 but quantum still wins) [There's graphs for this in the folder called vanguard_quantum_analysis-1]

Basket Size: min(8, max(5, n // 3)), where n is the actual number of bonds loaded
-- So if we have 20 bonds --> min (8, max(5, 6)) --> min(8,6) --> 6 basket size

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      |      6      | 0.0624    | 0.2698         | **0.1326**      |-42100  |
| **Classical**    |      6      | 0.0785    | 0.2792         | **1.1392**      |-42102  |

### Enhanced VQE Implementation

- **Circuit Architecture:** QAOA ansatz with 4 layers for optimal expressivity; Also utilizing LIGHTNING.QUBIT instead of DEFAULT.QUBIT (Using lightning.GPU in one of the examples and 3 ansatz layers and 100 batch size)
- **Multiple Restarts:** 4 restarts × 200 iterations each for robust optimization (6 restarts * 150 iterations with lightning.GPU)
- **Adative Learning Rate:** Use adaptive learning rate of 0.05 during optimization
- **Adaptive Penalties:** Dynamic constraint weight adjustment during optimization
- **Sampling Strategy:** 50,000 shots for high-precision measurements
- **Post-Processing:** Greedy feasibility correction for quantum solutions

## Scalability Analysis

- **Current Implementation:**
  - **16 qubits (20-25 qubits for lightning.GPU):** Substantial classical simulation requirements
  - **65,536+ possible solutions:** Classical enumeration becomes challenging
  - **Complexity:** Exponential classical search space vs polynomial VQE optimization

- **Scaling Potential:**
  - **Real-world portfolios:** 50-500+ assets (using vanguard data now!)
  - **Quantum advantage:** Expected at 20+ qubits where classical optimization becomes intractable (this is why you need lightning.gpu for more than 20 simulation qubits)
  - **Hybrid approaches:** Quantum sampling with classical post-processing proves effective (You can see more of this in the part 3 py file)

**Task 1 (Mathematical Review):**

- Identified binary decision variables, linear constraints, and quadratic objective
- Understanding of the portfolio optimization formulation from the challenge image

**Task 2 (Quantum Formulation):**

- Converted constrained problem to unconstrained using penalty methods
- Proper QUBO formulation with all constraints included

**Task 3 (Quantum Program):**

- Implemented VQE with QAOA ansatz (4 layers) and adaptive penalties
- Hamiltonian construction and optimization

**Task 4 (Solve with Quantum):**

- Optimized and found quantum solutions (see above section for enhanced VQE)
- Proper sampling and result interpretation

**Task 5 (Classical Comparison):**

- Implemented classical brute-force benchmark with simulated annealing with basin hopping algorithm
- Performance comparison with releveant metrics (see examples below)

## How to Run

1. Install dependencies:

   ```bash
   pip install pennylane matplotlib scipy
   ```

2. Run the script:

   ```bash
   python "exploring part 3 of step1-5.py"
   ```

3. Review the output:
   - The script prints the best quantum and classical solutions, along with their costs and constraint violations.
   - Top 10 quantum solutions with frequency and feasibility analysis
   - Comprehensive constraint satisfaction metrics

## Real Data Example 1 output for exploring part 3 of steps1-5.py (WITH Lightning.GPU (20 qubits))

```text
=== Hardware-Optimized Quantum Portfolio Optimization ===

=== Loading Real Vanguard Portfolio Data ===

Loading Vanguard VCIT bond portfolio data...
Successfully loaded 2629 bond positions from Vanguard portfolio
Dataset dimensions: 2629 assets x 278 features
Filtered to 2618 bond positions
Selected top 20 holdings by market value for optimization
  Returns calculated from OAS (credit spreads): 90 bps average
  Risk calculated from duration (avg: 5.85 years) and credit spreads
  Correlation matrix estimated from sector/credit clustering (avg: 0.303)
  Weights calculated from market values (largest: 0.134)

Portfolio Analysis Summary:
  Fund: VCIT ($2,665,535,685)
  Average Duration: 5.85 years
  Average Credit Spread: 90 basis points
  Expected Returns: [0.049, 0.074]
  Risk Measures: [0.044, 0.070]
  Sector Distribution: ['Financial', 'Industrial', 'Treasury Bond Portfolio']

=== Real Portfolio Characteristics ===
Fund: VCIT
Portfolio size: 20 bonds (quantum optimized)
Average duration: 5.85 years
Average credit spread: 90 basis points
Returns range: [0.049, 0.074]
Risk range: [0.044, 0.070]
Sample bonds: ['US91282CLN91', 'US87264ABF12', 'US06051GLH01']

=== Quantum Optimization Problem Setup ===
Real Vanguard bonds: 20, Target portfolio: 6
Data source: vanguard_real
Fund: VCIT ($2,665,535,685)

Real Market Parameters:
  Expected returns (m): [4.91, 7.35]%
  Return bounds (M): [5.89, 8.82]%
  Risk measures (i_c): [0.444, 0.699]
  Position weights (x_c): [1.475, 5.360]

Portfolio Constraints:
  Target basket size: 6 bonds
  Yield range: [3.0%, 6.0%]
  Duration target: 0.058
  Risk aversion: 2.0

Sample Real Bonds:
  US91282CLN91: Return=0.049, Risk=0.045
  US87264ABF12: Return=0.057, Risk=0.048
  US06051GLH01: Return=0.061, Risk=0.065
  US00287YBX67: Return=0.054, Risk=0.045
  US716973AE24: Return=0.059, Risk=0.068

Optimization ready: 20 real Vanguard bonds → 6 quantum portfolio

=== Building QUBO with Real Bond Data ===
Covariance matrix built from real bond correlations: (20, 20)
QUBO matrix Q shape: (20, 20)

=== Initializing Optimized Components ===
Hardware Config:
  CPUs: 16, Memory: 7.3GB
  Recommended: {'qaoa_layers': 3, 'shots': 36681, 'restarts': 6, 'processes': 4, 'batch_size': 100}
Main device: lightning.gpu
Circuit Analysis:
  Significant gates: 210
  Estimated memory: 16.01 MB

QAOA layers: 3, Restarts: 6

--- Restart 1/6 ---
  Iter   0: 3387.1966 | Cache: 1/500 entries, 0.0% hit rate
  Iter  50: 712.5424 | Cache: 51/500 entries, 0.0% hit rate
  Iter 100: -315.2314 | Cache: 101/500 entries, 0.0% hit rate
  Iter 150:  72.2990 | Cache: 151/500 entries, 0.0% hit rate
New best: 2410.7104

--- Restart 2/6 ---
  Iter   0: 2063.2704 | Cache: 201/500 entries, 0.0% hit rate
  Iter  50: -467.8108 | Cache: 251/500 entries, 0.0% hit rate
  Iter 100: -2610.6298 | Cache: 301/500 entries, 0.0% hit rate
  Iter 150: -1290.3852 | Cache: 351/500 entries, 0.0% hit rate
New best: -316.0350

--- Restart 3/6 ---
  Iter   0: -174.9646 | Cache: 401/500 entries, 0.0% hit rate
  Iter  50: -818.6676 | Cache: 451/500 entries, 0.0% hit rate
  Iter 100: 454.9084 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -310.2751 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 4/6 ---
  Iter   0: -838.9887 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -260.3688 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: 526.8209 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -559.2327 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 5/6 ---
  Iter   0: 317.6500 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -116.8064 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -96.2636 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: 2406.7156 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 6/6 ---
  Iter   0: -119.8970 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -153.6804 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -217.6202 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -315.2520 | Cache: 500/500 entries, 0.0% hit rate

=== Optimized Sampling ===
Sampling: 36681 shots in 0.29s

Top quantum solutions:
   1. Cost: -42357.21, Freq:   46
   2. Cost: -42357.21, Freq:   39
   3. Cost: -42359.65, Freq:   37
   4. Cost: -42340.07, Freq:   36
   5. Cost: -42315.32, Freq:   36
   6. Cost: -42359.65, Freq:   32
   7. Cost: -42357.21, Freq:   32
   8. Cost: -42357.21, Freq:   31
   9. Cost: -42359.65, Freq:   29
  10. Cost: -42340.98, Freq:   28

Classical benchmark...

=== Hardware-Optimized Performance Summary ===
Backend: lightning.gpu
Circuit gates: 210 (optimized)
Memory usage: 16.01 MB

Timing:
  Training: 1612.60s
  Sampling: 0.29s
  Classical: 0.12s
  Total quantum: 1612.90s

Results:
  Best quantum cost: -42359.6522
  Best classical cost: -42351.2775
  Quantum advantage: 1.000x

System Performance:
  Peak CPU: 0.0%
  Peak Memory: 3517.1 MB
  Duration: 1618.9s

Optimization Stats:
  Cache: 500/500 entries, 0.0% hit rate
  Best quantum cost: -42359.6522

Quantum Solution Analysis:
  Selected 6 bonds (target: 6):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US06051GMA49: Return=0.062, Risk=0.070
    3. US46647PDR47: Return=0.061, Risk=0.066
    4. US95000U3F88: Return=0.062, Risk=0.067
    5. US06051GLU12: Return=0.062, Risk=0.068
    6. US95000U3B74: Return=0.057, Risk=0.067
  Portfolio metrics:
    Cash flow: 0.0663 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2698 (target: [1.6 - 2.0])
    Average return: 0.061 (6.1%)
    Average risk: 0.067
    Total constraint violation: 1.3365

Classical Solution Analysis:
  Selected 7 bonds (target: 6):
    1. US06051GMA49: Return=0.062, Risk=0.070
    2. US06051GKY43: Return=0.060, Risk=0.062
    3. US097023CY98: Return=0.062, Risk=0.046
    4. US95000U3F88: Return=0.062, Risk=0.067
    5. US031162DR88: Return=0.060, Risk=0.066
    6. US172967MP39: Return=0.059, Risk=0.047
    7. US95000U3D31: Return=0.062, Risk=0.066
  Portfolio metrics:
    Cash flow: 0.0766 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2633 (target: [1.6 - 2.0])
    Average return: 0.061 (6.1%)
    Average risk: 0.061
    Total constraint violation: 2.3533

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Final Quantum Solution Analysis:
  Selected 6 bonds (target: 6):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US06051GMA49: Return=0.062, Risk=0.070
    3. US46647PDR47: Return=0.061, Risk=0.066
    4. US95000U3F88: Return=0.062, Risk=0.067
    5. US06051GLU12: Return=0.062, Risk=0.068
    6. US95000U3B74: Return=0.057, Risk=0.067
  Portfolio metrics:
    Cash flow: 0.0663 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2698 (target: [1.6 - 2.0])
    Average return: 0.061 (6.1%)
    Average risk: 0.067
    Total constraint violation: 1.3365

=== Real Data Integration Impact ===
Data source: vanguard_real (VCIT)
Real portfolio: $2,665,535,685 market value
Authentic bonds: 20 from 20 total positions
Real correlations: Avg 0.303
Actual risk-return: Duration 5.8y, Spread 90bp

=== Hardware Optimization Impact ===
Backend: lightning.gpu (hardware-optimized)
Circuit gates: 210 (memory-optimized)
Caching: Cache: 500/500 entries, 0.0% hit rate
Monitoring: Peak 3517MB
Configuration: 3 layers, 6 restarts

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -36428.23, Frequency = 46
Solution 2: Cost = -30471.71, Frequency = 39
Solution 3: Cost = -583.66, Frequency = 37
Solution 4: Cost = -30434.15, Frequency = 36
Solution 5: Cost = -40363.69, Frequency = 36
Solution 6: Cost = -12563.88, Frequency = 32
Solution 7: Cost = -36451.70, Frequency = 32
Solution 8: Cost = -36446.99, Frequency = 31
Solution 9: Cost = -22540.13, Frequency = 29
Solution 10: Cost = -30455.63, Frequency = 28

```

## Real Data Example 2 output for exploring part 3 of steps1-5.py (WITH Lightning.GPU (20 qubits))
```text
== Hardware-Optimized Quantum Portfolio Optimization ===

=== Loading Real Vanguard Portfolio Data ===
Loading Vanguard VCIT bond portfolio data...
Successfully loaded 2629 bond positions from Vanguard portfolio
Dataset dimensions: 2629 assets x 278 features
Filtered to 2618 bond positions
Selected top 20 holdings by market value for optimization
  Returns calculated from OAS (credit spreads): 90 bps average
  Risk calculated from duration (avg: 5.85 years) and credit spreads
  Correlation matrix estimated from sector/credit clustering (avg: 0.303)
  Weights calculated from market values (largest: 0.134)

Portfolio Analysis Summary:
  Fund: VCIT ($2,665,535,685)
  Average Duration: 5.85 years
  Average Credit Spread: 90 basis points
  Expected Returns: [0.049, 0.074]
  Risk Measures: [0.044, 0.070]
  Sector Distribution: ['Financial', 'Industrial', 'Treasury Bond Portfolio']

=== Real Portfolio Characteristics ===
Fund: VCIT
Portfolio size: 20 bonds (quantum optimized)
Average duration: 5.85 years
Average credit spread: 90 basis points
Returns range: [0.049, 0.074]
Risk range: [0.044, 0.070]
Sample bonds: ['US91282CLN91', 'US87264ABF12', 'US06051GLH01']

=== Quantum Optimization Problem Setup ===
Real Vanguard bonds: 20, Target portfolio: 6
Data source: vanguard_real
Fund: VCIT ($2,665,535,685)

Real Market Parameters:
  Expected returns (m): [4.91, 7.35]%
  Return bounds (M): [5.89, 8.82]%
  Risk measures (i_c): [0.444, 0.699]
  Position weights (x_c): [1.475, 5.360]

Portfolio Constraints:
  Target basket size: 6 bonds
  Yield range: [3.0%, 6.0%]
  Duration target: 0.058
  Risk aversion: 2.0

Sample Real Bonds:
  US91282CLN91: Return=0.049, Risk=0.045
  US87264ABF12: Return=0.057, Risk=0.048
  US06051GLH01: Return=0.061, Risk=0.065
  US00287YBX67: Return=0.054, Risk=0.045
  US716973AE24: Return=0.059, Risk=0.068

Optimization ready: 20 real Vanguard bonds → 6 quantum portfolio

=== Building QUBO with Real Bond Data ===
Covariance matrix built from real bond correlations: (20, 20)
QUBO matrix Q shape: (20, 20)

=== Initializing Optimized Components ===
Hardware Config:
  CPUs: 16, Memory: 7.3GB
  Recommended: {'qaoa_layers': 3, 'shots': 36681, 'restarts': 6, 'processes': 4, 'batch_size': 100}
Main device: lightning.gpu
Circuit Analysis:
  Significant gates: 210
  Estimated memory: 16.01 MB

=== Starting Optimized Training ===
QAOA layers: 3, Restarts: 6

--- Restart 1/6 ---
  Iter   0: -8617.0178 | Cache: 1/500 entries, 0.0% hit rate
  Iter  50: 1317.2524 | Cache: 51/500 entries, 0.0% hit rate
  Iter 100: 1304.5634 | Cache: 101/500 entries, 0.0% hit rate
  Iter 150: 1716.1425 | Cache: 151/500 entries, 0.0% hit rate
New best: -3464.7871

--- Restart 2/6 ---
  Iter   0: -1069.8705 | Cache: 201/500 entries, 0.0% hit rate
  Iter  50: -439.2024 | Cache: 251/500 entries, 0.0% hit rate
  Iter 100: 331.4856 | Cache: 301/500 entries, 0.0% hit rate
  Iter 150: 261.3003 | Cache: 351/500 entries, 0.0% hit rate

--- Restart 3/6 ---
  Iter   0: 393.2238 | Cache: 401/500 entries, 0.0% hit rate
  Iter  50: -1240.2969 | Cache: 451/500 entries, 0.0% hit rate
  Iter 100: -249.7027 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -464.9459 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 4/6 ---
  Iter   0: 141.7047 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: 126.1367 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -323.2413 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: 529.7106 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 5/6 ---
  Iter   0:  27.0196 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: 462.4843 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -46.0504 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -140.0843 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 6/6 ---
  Iter   0: -801.4223 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: 508.8128 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -144.8660 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -481.7243 | Cache: 500/500 entries, 0.0% hit rate

=== Optimized Sampling ===
Sampling: 36681 shots in 0.27s

Top quantum solutions:
   1. Cost: -42100.10, Freq:   34
   2. Cost: -42100.79, Freq:   33
   3. Cost: -42100.68, Freq:   33
   4. Cost: -42100.79, Freq:   30
   5. Cost: -42100.68, Freq:   27
   6. Cost: -42100.05, Freq:   25
   7. Cost: -42100.10, Freq:   22
   8. Cost: -42100.68, Freq:   21
   9. Cost: -42100.10, Freq:   19
  10. Cost: -42100.68, Freq:   19

Classical benchmark...

=== Hardware-Optimized Performance Summary ===
Backend: lightning.gpu
Circuit gates: 210 (optimized)
Memory usage: 16.01 MB

Timing:
  Training: 1604.53s
  Sampling: 0.27s
  Classical: 0.12s
  Total quantum: 1604.80s

Results:
  Best quantum cost: -42100.7901
  Best classical cost: -42102.4849
  Quantum advantage: 1.000x

System Performance:
  Peak CPU: 0.0%
  Peak Memory: 3575.3 MB
  Duration: 1609.7s

Optimization Stats:
  Cache: 500/500 entries, 0.0% hit rate
  Best quantum cost: -42100.7901

Quantum Solution Analysis:
  Selected 6 bonds (target: 6):
    1. US06051GMA49: Return=0.062, Risk=0.070
    2. US46647PDR47: Return=0.061, Risk=0.066
    3. US95000U3F88: Return=0.062, Risk=0.067
    4. US031162DR88: Return=0.060, Risk=0.066
    5. US95000U3B74: Return=0.057, Risk=0.067
    6. US95000U3D31: Return=0.062, Risk=0.066
  Portfolio metrics:
    Cash flow: 0.0624 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2698 (target: [0.4, 0.8])
    Average return: 0.060 (6.0%)
    Average risk: 0.067 (6.7%)
    Total constraint violation: 0.1326

Classical Solution Analysis:
  Selected 7 bonds (target: 6):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US097023CY98: Return=0.062, Risk=0.046
    4. US46647PDH64: Return=0.059, Risk=0.062
    5. US95000U3F88: Return=0.062, Risk=0.067
    6. US06051GLU12: Return=0.062, Risk=0.068
    7. US17327CAR43: Return=0.064, Risk=0.064
  Portfolio metrics:
    Cash flow: 0.0785 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2792 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.063 (6.3%)
    Total constraint violation: 1.1392

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Final Quantum Solution Analysis:
  Selected 6 bonds (target: 6):
    1. US06051GMA49: Return=0.062, Risk=0.070
    2. US46647PDR47: Return=0.061, Risk=0.066
    3. US95000U3F88: Return=0.062, Risk=0.067
    4. US031162DR88: Return=0.060, Risk=0.066
    5. US95000U3B74: Return=0.057, Risk=0.067
    6. US95000U3D31: Return=0.062, Risk=0.066
  Portfolio metrics:
    Cash flow: 0.0624 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2698 (target: [0.4, 0.8])
    Average return: 0.060 (6.0%)
    Average risk: 0.067 (6.7%)
    Total constraint violation: 0.1326

=== Real Data Integration Impact ===
Data source: vanguard_real (VCIT)
Real portfolio: $2,665,535,685 market value
Authentic bonds: 20 from 20 total positions
Real correlations: Avg 0.303
Actual risk-return: Duration 5.8y, Spread 90bp

=== Hardware Optimization Impact ===
Backend: lightning.gpu (hardware-optimized)
Circuit gates: 210 (memory-optimized)
Caching: Cache: 500/500 entries, 0.0% hit rate
Monitoring: Peak 3575MB
Configuration: 3 layers, 6 restarts

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = 29872.48, Frequency = 34
Solution 2: Cost = -12122.31, Frequency = 33
Solution 3: Cost = 13873.46, Frequency = 33
Solution 4: Cost = -123.77, Frequency = 30
Solution 5: Cost = 13872.26, Frequency = 27
Solution 6: Cost = -12119.02, Frequency = 25
Solution 7: Cost = 29872.02, Frequency = 22
Solution 8: Cost = 13873.14, Frequency = 21
Solution 9: Cost = 29872.75, Frequency = 19
Solution 10: Cost = 29872.96, Frequency = 19

=== Creating Real Vanguard Portfolio Analysis Plots ===
Data: VCIT with 20 real bonds
Saving plots to: vanguard_quantum_analysis/
  Saved: vanguard_quantum_analysis/20250806_015105_performance_dashboard.png
  Saved: vanguard_quantum_analysis/20250806_015106_top_solutions_analysis.png
  Saved: vanguard_quantum_analysis/20250806_015107_portfolio_composition.png
  Saved: vanguard_quantum_analysis/20250806_015108_constraint_satisfaction.png
  Saved: vanguard_quantum_analysis/20250806_015109_system_performance.png
  Saved: vanguard_quantum_analysis/20250806_015110_qubo_structure.png
  Saved: vanguard_quantum_analysis/20250806_015111_solution_quality.png

```

## Below are examples of not using the real data just synthetic data

## Example Output for exploring part 3 of steps1-5.py [WITH Lightning.GPU (20 qubits)]

```text
=== Hardware-Optimized Quantum Portfolio Optimization ===


Problem size: 20 bonds, Target basket: 10
Market parameters (m): [0.68727006 0.97535715 0.86599697 0.79932924 0.57800932 0.57799726
 0.52904181 0.93308807 0.80055751 0.85403629 0.51029225 0.98495493
 0.91622132 0.60616956 0.59091248 0.59170225 0.65212112 0.76237822
 0.71597251 0.64561457]

Market parameters (M): [1.61185289 1.13949386 1.29214465 1.36636184 1.45606998 1.78517596
 1.19967378 1.51423444 1.59241457 1.04645041 1.60754485 1.17052412
 1.06505159 1.94888554 1.96563203 1.80839735 1.30461377 1.09767211
 1.68423303 1.44015249]

Risk characteristics (i_c): [0.27322294 0.49710615 0.22063311 0.74559224 0.35526799 0.59751337
 0.38702665 0.51204081 0.52802617 0.31091267 0.78175078 0.66507969
 0.76369936 0.73689641 0.55873999 0.75312454 0.2530955  0.31758972
 0.22713637 0.3951982 ]

Bond amounts if selected (x_c): [1.87984804 3.53041631 1.25914562 3.1827627  2.19725148 1.85365908
 2.92917525 1.71679936 5.11701216 1.17729936 1.57990907 4.59654824
 8.21807866 1.57568581 1.50185981 1.71707986 1.10795421 4.16596882
 1.9374733  3.5559606 ]

Target basket size: 10

Cash flow range: [0.0100, 0.0700]
Characteristic range: [0.6, 1.6]
Problem: 20 bonds → 10 portfolio
Building QUBO formulation...
QUBO matrix Q shape: (20, 20)

=== Initializing Optimized Components ===
Hardware Config:
  CPUs: 16, Memory: 7.3GB
  Recommended: {'qaoa_layers': 3, 'shots': 36681, 'restarts': 6, 'processes': 4, 'batch_size': 100}
Main device: lightning.gpu
Circuit Analysis:
  Significant gates: 210
  Estimated memory: 16.01 MB

=== Starting Optimized Training ===
QAOA layers: 3, Restarts: 6

--- Restart 1/6 ---
/opt/venv/lib/python3.10/site-packages/pennylane/devices/preprocess.py:289: UserWarning: Differentiating with respect to the input parameters of LinearCombination is not supported with the adjoint differentiation method. Gradients are computed only with regards to the trainable parameters of the circuit.

 Mark the parameters of the measured observables as non-trainable to silence this warning.
  warnings.warn(
  Iter   0:  11.5241 | Cache: 1/500 entries, 0.0% hit rate
  Iter  50:  29.4934 | Cache: 51/500 entries, 0.0% hit rate
  Iter 100: -53.9448 | Cache: 101/500 entries, 0.0% hit rate
  Iter 150:  55.0550 | Cache: 151/500 entries, 0.0% hit rate
New best: 22.6160

--- Restart 2/6 ---
  Iter   0:  -4.1661 | Cache: 201/500 entries, 0.0% hit rate
  Iter  50: -24.4928 | Cache: 251/500 entries, 0.0% hit rate
  Iter 100: -102.6367 | Cache: 301/500 entries, 0.0% hit rate
  Iter 150: -40.6631 | Cache: 351/500 entries, 0.0% hit rate
New best: -84.3621

--- Restart 3/6 ---
  Iter   0: -44.7622 | Cache: 401/500 entries, 0.0% hit rate
  Iter  50: -68.1531 | Cache: 451/500 entries, 0.0% hit rate
  Iter 100:  55.1293 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:  39.8304 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 4/6 ---
  Iter   0:  67.2276 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -23.9908 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -16.1134 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:  63.1905 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 5/6 ---
  Iter   0: -110.3614 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50:  21.9691 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -18.1478 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:   9.8562 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 6/6 ---
  Iter   0:   9.5002 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -65.7129 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: 100.8758 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:  30.3898 | Cache: 500/500 entries, 0.0% hit rate

=== Optimized Sampling ===
Sampling: 36681 shots in 0.40s

Top quantum solutions:
   1. Cost: -217360.73, Freq:    4
   2. Cost: -219235.58, Freq:    3
   3. Cost: -218511.65, Freq:    3
   4. Cost: -220144.28, Freq:    3
   5. Cost: -218572.17, Freq:    3
   6. Cost: -219806.21, Freq:    3
   7. Cost: -216266.67, Freq:    3
   8. Cost: -220062.42, Freq:    3
   9. Cost: -220550.40, Freq:    3
  10. Cost: -218334.62, Freq:    3

Classical benchmark...

=== Hardware-Optimized Performance Summary ===
Backend: lightning.gpu
Circuit gates: 210 (optimized)
Memory usage: 16.01 MB

Timing:
  Training: 1709.42s
  Sampling: 0.40s
  Classical: 0.14s
  Total quantum: 1709.82s

Results:
  Best quantum cost: -220550.3965
  Best classical cost: -220360.7243
  Quantum advantage: 0.999x

System Performance:
  Peak CPU: 0.0%
  Peak Memory: 3290.2 MB
  Duration: 1712.5s

Optimization Stats:
  Cache: 500/500 entries, 0.0% hit rate
  Best quantum cost: -220550.3965

Quantum Solution:
  Portfolio: [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0459 (range: [0.0100, 0.0700])
  Characteristic: 1.3502 (range: [0.6000, 1.6000])
  Constraint violation: 0.0000

Classical Solution:
  Portfolio: [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  Basket size: 10 (target: 10)
  Cash flow: 0.0416 (range: [0.0100, 0.0700])
  Characteristic: 1.6062 (range: [0.6000, 1.6000])
  Constraint violation: 0.0062

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Optimized Quantum Solution:
  Portfolio: [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0459 (range: [0.0100, 0.0700])
  Characteristic: 1.3502 (range: [0.6000, 1.6000])
  Constraint violation: 0.0000

=== Optimization Impact ===
Hardware-optimized backend: lightning.gpu
Memory-optimized circuits: 210 gates
Smart caching: Cache: 500/500 entries, 0.0% hit rate
Real-time monitoring: Peak 3290MB
Adaptive configuration: 3 layers, 6 restarts

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -217360.73, Frequency = 4
Solution 2: Cost = -179587.08, Frequency = 3
Solution 3: Cost = -218511.65, Frequency = 3
Solution 4: Cost = -180389.36, Frequency = 3
Solution 5: Cost = -218572.17, Frequency = 3
Solution 6: Cost = -213207.97, Frequency = 3
Solution 7: Cost = -216266.67, Frequency = 3
Solution 8: Cost = -203921.04, Frequency = 3
Solution 9: Cost = -196635.26, Frequency = 3
Solution 10: Cost = -209672.49, Frequency = 3

Memory cleanup completed

```

## Example Output for Exploring part 3 of steps1-5.py [WITH Lightning.GPU (20 qubits). This time, no constraints but quantum wins due to better objective value]

```text
=== Hardware-Optimized Quantum Portfolio Optimization ===


Problem size: 20 bonds, Target basket: 10
Market parameters (m): [0.68727006 0.97535715 0.86599697 0.79932924 0.57800932 0.57799726
 0.52904181 0.93308807 0.80055751 0.85403629 0.51029225 0.98495493
 0.91622132 0.60616956 0.59091248 0.59170225 0.65212112 0.76237822
 0.71597251 0.64561457]

Market parameters (M): [1.61185289 1.13949386 1.29214465 1.36636184 1.45606998 1.78517596
 1.19967378 1.51423444 1.59241457 1.04645041 1.60754485 1.17052412
 1.06505159 1.94888554 1.96563203 1.80839735 1.30461377 1.09767211
 1.68423303 1.44015249]

Risk characteristics (i_c): [0.27322294 0.49710615 0.22063311 0.74559224 0.35526799 0.59751337
 0.38702665 0.51204081 0.52802617 0.31091267 0.78175078 0.66507969
 0.76369936 0.73689641 0.55873999 0.75312454 0.2530955  0.31758972
 0.22713637 0.3951982 ]

Bond amounts if selected (x_c): [1.87984804 3.53041631 1.25914562 3.1827627  2.19725148 1.85365908
 2.92917525 1.71679936 5.11701216 1.17729936 1.57990907 4.59654824
 8.21807866 1.57568581 1.50185981 1.71707986 1.10795421 4.16596882
 1.9374733  3.5559606 ]

Target basket size: 10

Cash flow range: [0.0100, 0.0700]
Characteristic range: [0.6, 2.0]
Problem: 20 bonds → 10 portfolio
Building QUBO formulation...
QUBO matrix Q shape: (20, 20)

=== Initializing Optimized Components ===
Hardware Config:
  CPUs: 16, Memory: 7.3GB
  Recommended: {'qaoa_layers': 3, 'shots': 36681, 'restarts': 6, 'processes': 4, 'batch_size': 100}
Main device: lightning.gpu
Circuit Analysis:
  Significant gates: 210
  Estimated memory: 16.01 MB

=== Starting Optimized Training ===
QAOA layers: 3, Restarts: 6

--- Restart 1/6 ---
/opt/venv/lib/python3.10/site-packages/pennylane/devices/preprocess.py:289: UserWarning: Differentiating with respect to the input parameters of LinearCombination is not supported with the adjoint differentiation method. Gradients are computed only with regards to the trainable parameters of the circuit.

 Mark the parameters of the measured observables as non-trainable to silence this warning.
  warnings.warn(
  Iter   0: -47.8407 | Cache: 1/500 entries, 0.0% hit rate
  Iter  50: 212.7926 | Cache: 51/500 entries, 0.0% hit rate
  Iter 100:  22.2078 | Cache: 101/500 entries, 0.0% hit rate
  Iter 150: 114.9366 | Cache: 151/500 entries, 0.0% hit rate
New best: 76.9454

--- Restart 2/6 ---
  Iter   0:  69.0022 | Cache: 201/500 entries, 0.0% hit rate
  Iter  50:   3.3165 | Cache: 251/500 entries, 0.0% hit rate
  Iter 100:  23.1491 | Cache: 301/500 entries, 0.0% hit rate
  Iter 150:  34.4970 | Cache: 351/500 entries, 0.0% hit rate
New best: -20.7745

--- Restart 3/6 ---
  Iter   0:  75.3648 | Cache: 401/500 entries, 0.0% hit rate
  Iter  50:   5.4424 | Cache: 451/500 entries, 0.0% hit rate
  Iter 100: -76.6303 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:  49.5589 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 4/6 ---
  Iter   0: -14.5229 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -42.1904 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100:  90.4458 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -85.2690 | Cache: 500/500 entries, 0.0% hit rate
New best: -104.1063

--- Restart 5/6 ---
  Iter   0:   0.3574 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: 166.3744 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100:  11.5567 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -139.3331 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 6/6 ---
  Iter   0: -28.9471 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50:  23.0191 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -50.4719 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:  77.1995 | Cache: 500/500 entries, 0.0% hit rate

=== Optimized Sampling ===
Sampling: 36681 shots in 0.29s

Top quantum solutions:
   1. Cost: -221695.52, Freq:    4
   2. Cost: -221781.43, Freq:    3
   3. Cost: -221703.90, Freq:    3
   4. Cost: -221276.87, Freq:    3
   5. Cost: -221385.46, Freq:    3
   6. Cost: -221559.27, Freq:    3
   7. Cost: -221462.10, Freq:    3
   8. Cost: -221492.90, Freq:    3
   9. Cost: -220099.92, Freq:    3
  10. Cost: -220761.08, Freq:    3

Classical benchmark...

=== Hardware-Optimized Performance Summary ===
Backend: lightning.gpu
Circuit gates: 210 (optimized)
Memory usage: 16.01 MB

Timing:
  Training: 1585.67s
  Sampling: 0.29s
  Classical: 0.15s
  Total quantum: 1585.96s

Results:
  Best quantum cost: -221781.4298
  Best classical cost: -221776.7360
  Quantum advantage: 1.000x

System Performance:
  Peak CPU: 0.0%
  Peak Memory: 3524.1 MB
  Duration: 1588.8s

Optimization Stats:
  Cache: 500/500 entries, 0.0% hit rate
  Best quantum cost: -221781.4298

Quantum Solution:
  Portfolio: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0416 (range: [0.0100, 0.0700])
  Characteristic: 1.7418 (range: [1.6000, 2.0000])
  Constraint violation: 0.0000

Classical Solution:
  Portfolio: [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0368 (range: [0.0100, 0.0700])
  Characteristic: 1.6521 (range: [1.6000, 2.0000])
  Constraint violation: 0.0000

=== Final Assessment ===
QUANTUM ADVANTAGE: Better objective value

Optimized Quantum Solution:
  Portfolio: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0416 (range: [0.0100, 0.0700])
  Characteristic: 1.7418 (range: [1.6000, 2.0000])
  Constraint violation: 0.0000

=== Optimization Impact ===
Hardware-optimized backend: lightning.gpu
Memory-optimized circuits: 210 gates
Smart caching: Cache: 500/500 entries, 0.0% hit rate
Real-time monitoring: Peak 3524MB
Adaptive configuration: 3 layers, 6 restarts

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -207413.06, Frequency = 4
Solution 2: Cost = -208219.93, Frequency = 3
Solution 3: Cost = -177148.23, Frequency = 3
Solution 4: Cost = -214376.49, Frequency = 3
Solution 5: Cost = -209422.62, Frequency = 3
Solution 6: Cost = -206738.44, Frequency = 3
Solution 7: Cost = -215551.34, Frequency = 3
Solution 8: Cost = -215568.93, Frequency = 3
Solution 9: Cost = -220099.92, Frequency = 3
Solution 10: Cost = -181495.40, Frequency = 3

```

## Without GPU from here on out

## Example Output for exploring part 2 of steps1-5.py

```text
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

## Example output for exploring part 2 of steps1-5.py [THIS IS WITH 20 bonds (qubits) and 10 target basket size]

```text
=== Exploring (part 2) of Quantum Portfolio Optimization... ===
Problem size: 20 bonds, Target basket: 10
Market parameters (m): [0.68727006 0.97535715 0.86599697 0.79932924 0.57800932 0.57799726
 0.52904181 0.93308807 0.80055751 0.85403629 0.51029225 0.98495493
 0.91622132 0.60616956 0.59091248 0.59170225 0.65212112 0.76237822
 0.71597251 0.64561457]

Market parameters (M): [1.61185289 1.13949386 1.29214465 1.36636184 1.45606998 1.78517596
 1.19967378 1.51423444 1.59241457 1.04645041 1.60754485 1.17052412
 1.06505159 1.94888554 1.96563203 1.80839735 1.30461377 1.09767211
 1.68423303 1.44015249]

Risk characteristics (i_c): [0.27322294 0.49710615 0.22063311 0.74559224 0.35526799 0.59751337
 0.38702665 0.51204081 0.52802617 0.31091267 0.78175078 0.66507969
 0.76369936 0.73689641 0.55873999 0.75312454 0.2530955  0.31758972
 0.22713637 0.3951982 ]

Bond amounts if selected (x_c): [1.87984804 3.53041631 1.25914562 3.1827627  2.19725148 1.85365908
 2.92917525 1.71679936 5.11701216 1.17729936 1.57990907 4.59654824
 8.21807866 1.57568581 1.50185981 1.71707986 1.10795421 4.16596882
 1.9374733  3.5559606 ]

Target basket size: 10

Cash flow range: [0.0100, 0.0500]
Characteristic range: [0.6, 1.2]
Building QUBO formulation...
QUBO matrix Q shape: (20, 20)
Building Optimized Hamiltonian...
Hamiltonian constructed with 231 terms
Using lightning.qubit: Optimized CPU
Main computation device: lightning.qubit

Starting Optimized-Accelerated VQE training...

--- Restart 1/4 ---
radients are computed only with regards to the trainable parameters of the circuit.

 Mark the parameters of the measured observables as non-trainable to silence this warning.
  warnings.warn(
 Iteration   0, Cost: -305316.1333
 Iteration  50, Cost: -305197.4388
 Iteration 100, Cost: -305236.1765
 Iteration 150, Cost: -305265.5892
 Final cost: -305441.7030

--- Restart 2/4 ---
 Iteration   0, Cost: -305218.1815
 Iteration  50, Cost: -305257.5228
 Iteration 100, Cost: -305257.2086
 Iteration 150, Cost: -305293.2347
 Final cost: -305170.7953

--- Restart 3/4 ---
 Iteration   0, Cost: -305261.9120
 Iteration  50, Cost: -305448.0162
 Iteration 100, Cost: -305199.9361
 Iteration 150, Cost: -305245.2792
 Final cost: -305101.4352

--- Restart 4/4 ---
 Iteration   0, Cost: -305048.5286
 Iteration  50, Cost: -305340.2521
 Iteration 100, Cost: -305150.5672
 Iteration 150, Cost: -305294.2979
 Final cost: -305161.9966

Training completed in 8124.41 seconds
Best VQE cost: -305441.7030

Optimized-accelerated sampling...
Using lightning.qubit: Optimized CPU
Sampling device: lightning.qubit
Sampling completed in 1.43 seconds

Top quantum solutions (post-processed):
  1. [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0] (cost: -217640.67, freq:    5)
  2. [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1] (cost: -217959.35, freq:    4)
  3. [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1] (cost: -219746.85, freq:    3)
  4. [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0] (cost: -218895.36, freq:    3)
  5. [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1] (cost: -218688.03, freq:    3)
  6. [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1] (cost: -219898.97, freq:    3)
  7. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1] (cost: -215849.12, freq:    3)
  8. [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1] (cost: -217369.29, freq:    3)
  9. [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1] (cost: -219699.86, freq:    3)
 10. [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1] (cost: -220171.94, freq:    3)

Classical benchmark...

=== Optimized Performance Summary (with lightning) ===
Hardware: lightning.qubit
Problem size: 20 bonds, Target basket: 10
QAOA layers: 4, Restarts: 4

Timing:
  Training: 8124.41s
  Sampling: 1.43s
  Classical: 0.11s
  Total quantum: 8125.85s

Results:
  Best quantum cost: -220171.9435
  Best classical cost: -219163.5757
  Quantum advantage: 0.995x

Quantum Solution:
  Portfolio: [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0405 (range: [0.0100, 0.0500])
  Characteristic: 1.3335 (range: [0.6000, 1.2000])
  Constraint violation: 0.1335

Classical Solution:
  Portfolio: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  Basket size: 11 (target: 10)
  Cash flow: 0.0487 (range: [0.0100, 0.0500])
  Characteristic: 1.9542 (range: [0.6000, 1.2000])
  Constraint violation: 1.7542

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -213659.85, Frequency = 5
Solution 2: Cost = -217959.35, Frequency = 4
Solution 3: Cost = -136198.52, Frequency = 3
Solution 4: Cost = -216920.27, Frequency = 3
Solution 5: Cost = -206891.86, Frequency = 3
Solution 6: Cost = -213667.66, Frequency = 3
Solution 7: Cost = -215849.12, Frequency = 3
Solution 8: Cost = -217369.29, Frequency = 3
Solution 9: Cost = -219024.24, Frequency = 3
Solution 10: Cost = -196294.18, Frequency = 3
```

## Example Output for exploring part 2 of steps1-5.py (This time we increase the ranges for cash flow and characteristic)

```text
=== Exploring (part 2) of Quantum Portfolio Optimization... ===
Problem size: 20 bonds, Target basket: 10
Market parameters (m): [0.68727006 0.97535715 0.86599697 0.79932924 0.57800932 0.57799726
 0.52904181 0.93308807 0.80055751 0.85403629 0.51029225 0.98495493
 0.91622132 0.60616956 0.59091248 0.59170225 0.65212112 0.76237822
 0.71597251 0.64561457]

Market parameters (M): [1.61185289 1.13949386 1.29214465 1.36636184 1.45606998 1.78517596
 1.19967378 1.51423444 1.59241457 1.04645041 1.60754485 1.17052412
 1.06505159 1.94888554 1.96563203 1.80839735 1.30461377 1.09767211
 1.68423303 1.44015249]

Risk characteristics (i_c): [0.27322294 0.49710615 0.22063311 0.74559224 0.35526799 0.59751337
 0.38702665 0.51204081 0.52802617 0.31091267 0.78175078 0.66507969
 0.76369936 0.73689641 0.55873999 0.75312454 0.2530955  0.31758972
 0.22713637 0.3951982 ]

Bond amounts if selected (x_c): [1.87984804 3.53041631 1.25914562 3.1827627  2.19725148 1.85365908
 2.92917525 1.71679936 5.11701216 1.17729936 1.57990907 4.59654824
 8.21807866 1.57568581 1.50185981 1.71707986 1.10795421 4.16596882
 1.9374733  3.5559606 ]

Target basket size: 10

Cash flow range: [0.0100, 0.0700]
Characteristic range: [0.6, 1.6]
Building QUBO formulation...
QUBO matrix Q shape: (20, 20)
Building Optimized Hamiltonian...
Hamiltonian constructed with 231 terms
Using lightning.qubit: Optimized CPU
Main computation device: lightning.qubit

Starting Optimized-Accelerated VQE training...

--- Restart 1/4 ---
radients are computed only with regards to the trainable parameters of the circuit.

 Mark the parameters of the measured observables as non-trainable to silence this warning.
  warnings.warn(
Iteration   0, Cost: -305904.8654
 Iteration  50, Cost: -305694.9734
 Iteration 100, Cost: -305911.9845
 Iteration 150, Cost: -305870.0647
 Final cost: -305800.2998

--- Restart 2/4 ---
 Iteration   0, Cost: -305928.0308
 Iteration  50, Cost: -305870.0161
 Iteration 100, Cost: -305721.8220
 Iteration 150, Cost: -305891.6850
 Final cost: -305825.7073

--- Restart 3/4 ---
 Iteration   0, Cost: -305886.4285
 Iteration  50, Cost: -305891.2708
 Iteration 100, Cost: -305686.4281
 Iteration 150, Cost: -305844.9364
 Final cost: -305957.4939

--- Restart 4/4 ---
 Iteration   0, Cost: -305866.9482
 Iteration  50, Cost: -305753.4372
 Iteration 100, Cost: -305849.4151
 Iteration 150, Cost: -305924.3343
 Final cost: -304715.7024

Training completed in 7987.35 seconds
Best VQE cost: -305957.4939

Optimized-accelerated sampling...
Using lightning.qubit: Optimized CPU
Sampling device: lightning.qubit
Sampling completed in 1.27 seconds

Top quantum solutions (post-processed):
  1. [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1] (cost: -218947.17, freq:    4)
  2. [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1] (cost: -220570.18, freq:    4)
  3. [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0] (cost: -219759.18, freq:    4)
  4. [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1] (cost: -220470.70, freq:    4)
  5. [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1] (cost: -219900.91, freq:    4)
  6. [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] (cost: -220171.90, freq:    3)
  7. [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1] (cost: -218246.79, freq:    3)
  8. [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1] (cost: -220341.23, freq:    3)
  9. [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1] (cost: -218402.34, freq:    3)
 10. [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0] (cost: -219408.03, freq:    3)

Classical benchmark...

=== Optimized Performance Summary (with lightning) ===
Hardware: lightning.qubit
Problem size: 20 bonds, Target basket: 10
QAOA layers: 4, Restarts: 4

Timing:
  Training: 7987.35s
  Sampling: 1.27s
  Classical: 0.11s
  Total quantum: 7988.62s

Results:
  Best quantum cost: -220570.1805
  Best classical cost: -220112.0406
  Quantum advantage: 0.998x

Quantum Solution:
  Portfolio: [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0417 (range: [0.0100, 0.0700])
  Characteristic: 1.3150 (range: [0.6000, 1.6000])
  Constraint violation: 0.0000

Classical Solution:
  Portfolio: [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  Basket size: 10 (target: 10)
  Cash flow: 0.0464 (range: [0.0100, 0.0700])
  Characteristic: 1.7846 (range: [0.6000, 1.6000])
  Constraint violation: 0.1846

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -207074.25, Frequency = 4
Solution 2: Cost = -153208.36, Frequency = 4
Solution 3: Cost = -203917.21, Frequency = 4
Solution 4: Cost = -208561.26, Frequency = 4
Solution 5: Cost = -190803.79, Frequency = 4
Solution 6: Cost = -191815.69, Frequency = 3
Solution 7: Cost = -218246.79, Frequency = 3
Solution 8: Cost = -213727.83, Frequency = 3
Solution 9: Cost = -218402.34, Frequency = 3
Solution 10: Cost = -207506.40, Frequency = 3
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

- Quantum Approximate Optimization Algorithm with 4 layers with lightning.qubit (3 layers with lightning.gpu and shots are 36681 with 6 restarts, 4 processes, and batch size of 100)
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
- Includes 7 plots under the current_progress/quantum_analysis_plots

### **Algorithm Performance:**

**Strengths:**

- VQE shows good convergence behavior
- Quantum algorithm explores different solution regions
- Handles multiple constraints simultaneously
- Proper constraint violation analysis

**Areas for Improvement:**

- Penalty parameter tuning needed for better constraint satisfaction
- Larger problem sizes would better demonstrate quantum advantage, in theory, however it will be harder due to limited amount of qubits in the world

## Technical Implementation

- **Quantum Framework:** PennyLane with lightning.qubit and lightning.gpu simulators
- **Optimization:** Adam optimizer with adaptive learning rate of 0.05 and 0.01
- **Sampling:** 50,000 shots for high-precision measurements (36681 shots for lightning.gpu)
- **Classical Benchmark:** Simulated annealing with basin hopping
- **Constraint Handling:** Penalty method with adaptive scaling

## References

- [The Wiser's Quantum Portfolio Optimization](https://www.thewiser.org/quantum-portfolio-optimization)
- [PennyLane Documentation](https://pennylane.ai/)
