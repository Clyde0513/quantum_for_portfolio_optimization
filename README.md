# Quantum Portfolio Optimization

(**for exploring part 3 of step1-5.py**)

This project demonstrates a quantum approach to portfolio optimization using Variational Quantum Eigensolver (VQE) with QAOA ansatz and PennyLane. The implementation achieves competitive performance with classical methods while demonstrating  constraint satisfaction capabilities. This is a Hackathon for Wiser's Project 2: Quantum for Portfolio Optimization.

## Overview

The goal of this project is to solve a Quadratic Unconstrained Binary Optimization (QUBO) problem for portfolio optimization. The problem is mapped to a quantum Hamiltonian, and the ground state of the Hamiltonian represents the optimal portfolio.

### Key Results

## Real Vanguard Data Example 1 (3 QAOA layers deep)

### Constraint Analysis of 20 bonds, basket size 6 (Some constraints because we made the characteristic ranges from 1.6 to 2.0 but quantum still wins) [There's graphs for this in the folder called vanguard_quantum_analysis]

## In the next run, we will make the characteristic range realistic now, just experimenting with it

Basket Size: min(8, max(5, n // 3)), where n is the actual number of bonds loaded
-- So if we have 20 bonds --> min (8, max(5, 6)) --> min(8,6) --> 6 basket size

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      |      6      | 0.0663    | 0.2698         | **1.3365**      |-42359  |
| **Classical**    |      6      | 0.0766    | 0.2633         | **2.3533**      |-42351  |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Real Vanguard Data Example 2 (3 QAOA layers deep)

### Constraint Analysis of 20 bonds, basket size 6 (Characteristic ranges from 0.4 to 0.8 but quantum still wins) [There's graphs for this in the folder called vanguard_quantum_analysis-1]

Basket Size: min(8, max(5, n // 3)), where n is the actual number of bonds loaded
-- So if we have 20 bonds --> min (8, max(5, 6)) --> min(8,6) --> 6 basket size

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      |      6      | 0.0624    | 0.2698         | **0.1326**      |-42100  |
| **Classical**    |      6      | 0.0785    | 0.2792         | **1.1392**      |-42102  |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Real Vanguard Data Example 3

### Constraint Analysis of 15 bonds, basket size 5 (3 QAOA layers deep)

Basket Size: min(8, max(5, n // 3)), where n is the actual number of bonds loaded
-- So if we have 15 bonds --> min (8, max(5, 5)) --> min(8,5) --> 5 basket size

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      |      5      | 0.0546    | 0.2276         | **0.1724**      |-30089  |
| **Classical**    |      5      | 0.0653    | 0.2623         | **1.1430**      |-30098  |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Real Vanguard Data Example 4

### Constraint Analysis of 15 bonds, basket size 5  (15 QAOA layers deep) (Graphs under vanguard_quantum_analysis-3 folder)

Basket Size: min(8, max(5, n // 3)), where n is the actual number of bonds loaded
-- So if we have 15 bonds --> min (8, max(5, 5)) --> min(8,5) --> 5 basket size

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      |      5      | 0.0555    | 0.2254         | **0.1746**      |-30088  |
| **Classical**    |      5      | 0.0633    | 0.2394         | **1.1639**      |-30092  |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Real Vanguard Data Example 5

### Constraint Analysis of 24 bonds, basket size 8 (2 QAOA layers deep) (Graphs under vanguard_quantum_analysis-4 folder)

Basket Size: min(8, max(5, n // 3)), where n is the actual number of bonds loaded
-- So if we have 24 bonds --> min (8, max(5, 8)) --> min(8,8) --> 8 basket size

| Solution         | Basket Size | Cash Flow | Characteristic | Total Violation | Cost   |
|------------------|-------------|-----------|----------------|-----------------|------- |
| **Quantum**      |      8      | 0.0900    | 0.3478         | **0.0822**      |-72117  |
| **Classical**    |    9/8      | 0.0955    | 0.3793         | **1.0562**      |-72122  |

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
  - **Quantum advantage:** Expected at 20+ qubits where classical optimization becomes intractable
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

## Real Data Example 3 output for exploring part 3 of steps1-5.py (With Lightning.GPU (15 qubits))

```text
== Hardware-Optimized Quantum Portfolio Optimization ===

=== Loading Real Vanguard Portfolio Data ===
Loading Vanguard VCIT bond portfolio data...
Successfully loaded 2629 bond positions from Vanguard portfolio
Dataset dimensions: 2629 assets x 278 features
Filtered to 2618 bond positions
Selected top 15 holdings by market value for optimization
  Returns calculated from OAS (credit spreads): 91 bps average
  Risk calculated from duration (avg: 5.89 years) and credit spreads
  Correlation matrix estimated from sector/credit clustering (avg: 0.303)
  Weights calculated from market values (largest: 0.166)

Portfolio Analysis Summary:
  Fund: VCIT ($2,153,890,208)
  Average Duration: 5.89 years
  Average Credit Spread: 91 basis points
  Expected Returns: [0.049, 0.074]
  Risk Measures: [0.045, 0.070]
  Sector Distribution: ['Financial', 'Industrial', 'Treasury Bond Portfolio']

=== Real Portfolio Characteristics ===
Fund: VCIT
Portfolio size: 15 bonds (quantum optimized)
Average duration: 5.89 years
Average credit spread: 91 basis points
Returns range: [0.049, 0.074]
Risk range: [0.045, 0.070]
Sample bonds: ['US91282CLN91', 'US87264ABF12', 'US06051GLH01']

=== Quantum Optimization Problem Setup ===
Real Vanguard bonds: 15, Target portfolio: 5
Data source: vanguard_real
Fund: VCIT ($2,153,890,208)

Real Market Parameters:
  Expected returns (m): [4.91, 7.35]%
  Return bounds (M): [5.89, 8.82]%
  Risk measures (i_c): [0.447, 0.699]
  Position weights (x_c): [1.525, 4.975]

Portfolio Constraints:
  Target basket size: 5 bonds
  Yield range: [3.0%, 6.0%]
  Duration target: 0.059
  Risk aversion: 2.0

Sample Real Bonds:
  US91282CLN91: Return=0.049, Risk=0.045
  US87264ABF12: Return=0.057, Risk=0.048
  US06051GLH01: Return=0.061, Risk=0.065
  US00287YBX67: Return=0.054, Risk=0.045
  US716973AE24: Return=0.059, Risk=0.068

Optimization ready: 15 real Vanguard bonds → 5 quantum portfolio

=== Building QUBO with Real Bond Data ===
Covariance matrix built from real bond correlations: (15, 15)
QUBO matrix Q shape: (15, 15)

=== Initializing Optimized Components ===
Hardware Config:
  CPUs: 16, Memory: 7.3GB
  Recommended: {'qaoa_layers': 3, 'shots': 36681, 'restarts': 5, 'processes': 4, 'batch_size': 312}
Main device: lightning.gpu
Circuit Analysis:
  Significant gates: 120
  Estimated memory: 0.51 MB

=== Starting Optimized Training ===
QAOA layers: 3, Restarts: 5

--- Restart 1/5 ---
  Iter   0: 2235.7510 | Cache: 1/500 entries, 0.0% hit rate
  Iter  50: 646.2934 | Cache: 51/500 entries, 0.0% hit rate
  Iter 100: 284.5512 | Cache: 101/500 entries, 0.0% hit rate
  Iter 150: 1421.9041 | Cache: 151/500 entries, 0.0% hit rate
New best: -147.8232

--- Restart 2/5 ---
  Iter   0: 333.5096 | Cache: 201/500 entries, 0.0% hit rate
  Iter  50: -214.2464 | Cache: 251/500 entries, 0.0% hit rate
  Iter 100: 116.7286 | Cache: 301/500 entries, 0.0% hit rate
  Iter 150: -207.4894 | Cache: 351/500 entries, 0.0% hit rate
New best: -1609.3758

--- Restart 3/5 ---
  Iter   0: 617.4431 | Cache: 401/500 entries, 0.0% hit rate
  Iter  50: -404.4405 | Cache: 451/500 entries, 0.0% hit rate
  Iter 100: -641.2667 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:  80.8913 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 4/5 ---
  Iter   0: -397.5769 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -208.9377 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -544.1584 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -323.2236 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 5/5 ---
  Iter   0: -190.2146 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -2295.5743 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -42.8319 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: 173.8719 | Cache: 500/500 entries, 0.0% hit rate

=== Optimized Sampling ===
Sampling: 36681 shots in 0.13s

Top quantum solutions:
   1. Cost: -30087.48, Freq:  196
   2. Cost: -30087.57, Freq:  191
   3. Cost: -30087.57, Freq:  150
   4. Cost: -30082.17, Freq:  123
   5. Cost: -30088.93, Freq:  122
   6. Cost: -30082.30, Freq:  110
   7. Cost: -30089.25, Freq:  108
   8. Cost: -30081.91, Freq:  105
   9. Cost: -30072.90, Freq:  101
  10. Cost: -30077.22, Freq:   98

Classical benchmark...

=== Hardware-Optimized Performance Summary ===
Backend: lightning.gpu
Circuit gates: 120 (optimized)
Memory usage: 0.51 MB

Timing:
  Training: 408.27s
  Sampling: 0.13s
  Classical: 0.10s
  Total quantum: 408.40s

Results:
  Best quantum cost: -30089.2520
  Best classical cost: -30098.7602
  Quantum advantage: 1.000x

System Performance:
  Peak CPU: 0.0%
  Peak Memory: 2300.1 MB
  Duration: 415.7s

Optimization Stats:
  Cache: 500/500 entries, 0.0% hit rate
  Best quantum cost: -30089.2520

Quantum Solution Analysis:
  Selected 5 bonds (target: 5):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US06051GMA49: Return=0.062, Risk=0.070
    4. US95000U3F88: Return=0.062, Risk=0.067
    5. US06051GLU12: Return=0.062, Risk=0.068
  Portfolio metrics:
    Cash flow: 0.0546 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2276 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.067 (6.7%)
    Total constraint violation: 0.1724

Classical Solution Analysis:
  Selected 6 bonds (target: 5):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US06051GMA49: Return=0.062, Risk=0.070
    3. US46647PDR47: Return=0.061, Risk=0.066
    4. US55903VBC63: Return=0.074, Risk=0.062
    5. US95000U3F88: Return=0.062, Risk=0.067
    6. US031162DR88: Return=0.060, Risk=0.066
  Portfolio metrics:
    Cash flow: 0.0653 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2623 (target: [0.4, 0.8])
    Average return: 0.063 (6.3%)
    Average risk: 0.066 (6.6%)
    Total constraint violation: 1.1430

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Final Quantum Solution Analysis:
  Selected 5 bonds (target: 5):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US06051GMA49: Return=0.062, Risk=0.070
    4. US95000U3F88: Return=0.062, Risk=0.067
    5. US06051GLU12: Return=0.062, Risk=0.068
  Portfolio metrics:
    Cash flow: 0.0546 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2276 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.067 (6.7%)
    Total constraint violation: 0.1724

=== Real Data Integration Impact ===
Data source: vanguard_real (VCIT)
Real portfolio: $2,153,890,208 market value
Authentic bonds: 15 from 15 total positions
Real correlations: Avg 0.303
Actual risk-return: Duration 5.9y, Spread 91bp

=== Hardware Optimization Impact ===
Backend: lightning.gpu (hardware-optimized)
Circuit gates: 120 (memory-optimized)
Caching: Cache: 500/500 entries, 0.0% hit rate
Monitoring: Peak 2300MB
Configuration: 3 layers, 5 restarts

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -10116.73, Frequency = 196
Solution 2: Cost = -24107.83, Frequency = 191
Solution 3: Cost = -18115.20, Frequency = 150
Solution 4: Cost = -28065.98, Frequency = 123
Solution 5: Cost = -24108.95, Frequency = 122
Solution 6: Cost = -28092.91, Frequency = 110
Solution 7: Cost = -10122.46, Frequency = 108
Solution 8: Cost = -28092.67, Frequency = 105
Solution 9: Cost = -30072.90, Frequency = 101
Solution 10: Cost = -30083.41, Frequency = 98

=== Creating Real Vanguard Portfolio Analysis Plots ===
Data: VCIT with 15 real bonds
Saving plots to: vanguard_quantum_analysis-2/
  Saved: vanguard_quantum_analysis-2/20250806_150442_performance_dashboard.png
  Saved: vanguard_quantum_analysis-2/20250806_150443_top_solutions_analysis.png
  Saved: vanguard_quantum_analysis-2/20250806_150443_portfolio_composition.png
  Saved: vanguard_quantum_analysis-2/20250806_150445_constraint_satisfaction.png
  Saved: vanguard_quantum_analysis-2/20250806_150446_system_performance.png
  Saved: vanguard_quantum_analysis-2/20250806_150447_qubo_structure.png
  Saved: vanguard_quantum_analysis-2/20250806_150448_solution_quality.png
```

## Real Data Example 4 output for exploring part 3 of steps1-5.py (With Lightning.GPU (15 qubits) but this time we tried increasing the QAOA Layers to 15 deep)

```text
=== Hardware-Optimized Quantum Portfolio Optimization ===

=== Loading Real Vanguard Portfolio Data ===
Loading Vanguard VCIT bond portfolio data...
Successfully loaded 2629 bond positions from Vanguard portfolio
Dataset dimensions: 2629 assets x 278 features
Filtered to 2618 bond positions
Selected top 15 holdings by market value for optimization
  Returns calculated from OAS (credit spreads): 91 bps average
  Risk calculated from duration (avg: 5.89 years) and credit spreads
  Correlation matrix estimated from sector/credit clustering (avg: 0.303)
  Weights calculated from market values (largest: 0.166)

Portfolio Analysis Summary:
  Fund: VCIT ($2,153,890,208)
  Average Duration: 5.89 years
  Average Credit Spread: 91 basis points
  Expected Returns: [0.049, 0.074]
  Risk Measures: [0.045, 0.070]
  Sector Distribution: ['Financial', 'Industrial', 'Treasury Bond Portfolio']

=== Real Portfolio Characteristics ===
Fund: VCIT
Portfolio size: 15 bonds (quantum optimized)
Average duration: 5.89 years
Average credit spread: 91 basis points
Returns range: [0.049, 0.074]
Risk range: [0.045, 0.070]
Sample bonds: ['US91282CLN91', 'US87264ABF12', 'US06051GLH01']

=== Quantum Optimization Problem Setup ===
Real Vanguard bonds: 15, Target portfolio: 5
Data source: vanguard_real
Fund: VCIT ($2,153,890,208)

Real Market Parameters:
  Expected returns (m): [4.91, 7.35]%
  Return bounds (M): [5.89, 8.82]%
  Risk measures (i_c): [0.447, 0.699]
  Position weights (x_c): [1.525, 4.975]

Portfolio Constraints:
  Target basket size: 5 bonds
  Yield range: [3.0%, 6.0%]
  Duration target: 0.059
  Risk aversion: 2.0

Sample Real Bonds:
  US91282CLN91: Return=0.049, Risk=0.045
  US87264ABF12: Return=0.057, Risk=0.048
  US06051GLH01: Return=0.061, Risk=0.065
  US00287YBX67: Return=0.054, Risk=0.045
  US716973AE24: Return=0.059, Risk=0.068

Optimization ready: 15 real Vanguard bonds → 5 quantum portfolio

=== Building QUBO with Real Bond Data ===
Covariance matrix built from real bond correlations: (15, 15)
QUBO matrix Q shape: (15, 15)

=== Initializing Optimized Components ===
Hardware Config:
  CPUs: 16, Memory: 7.3GB
  Recommended: {'qaoa_layers': 15, 'shots': 36681, 'restarts': 5, 'processes': 4, 'batch_size': 312}
Main device: lightning.gpu
Circuit Analysis:
  Significant gates: 120
  Estimated memory: 0.51 MB

=== Starting Optimized Training ===
QAOA layers: 15, Restarts: 5

--- Restart 1/5 ---
  Iter   0:   7.7829 | Cache: 1/500 entries, 0.0% hit rate
  Iter  50: -49.8649 | Cache: 51/500 entries, 0.0% hit rate
  Iter 100:  26.9868 | Cache: 101/500 entries, 0.0% hit rate
  Iter 150: -165.9924 | Cache: 151/500 entries, 0.0% hit rate
New best: -83.5176

--- Restart 2/5 ---
  Iter   0:  78.2687 | Cache: 201/500 entries, 0.0% hit rate
  Iter  50: 221.9504 | Cache: 251/500 entries, 0.0% hit rate
  Iter 100:  19.3818 | Cache: 301/500 entries, 0.0% hit rate
  Iter 150:  -2.7413 | Cache: 351/500 entries, 0.0% hit rate

--- Restart 3/5 ---
  Iter   0: -17.6540 | Cache: 401/500 entries, 0.0% hit rate
  Iter  50: 144.3433 | Cache: 451/500 entries, 0.0% hit rate
  Iter 100: -148.7635 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:  94.4706 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 4/5 ---
  Iter   0: -185.8001 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -55.0280 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: 139.2483 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150:   5.7107 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 5/5 ---
  Iter   0:  97.8533 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -91.7477 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: 133.6604 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -77.2422 | Cache: 500/500 entries, 0.0% hit rate

=== Optimized Sampling ===
Sampling: 36681 shots in 0.70s

Top quantum solutions:
   1. Cost: -30086.55, Freq:   15
   2. Cost: -30081.27, Freq:   14
   3. Cost: -30084.67, Freq:   14
   4. Cost: -30087.48, Freq:   14
   5. Cost: -30085.61, Freq:   14
   6. Cost: -30088.01, Freq:   13
   7. Cost: -30086.93, Freq:   13
   8. Cost: -30086.85, Freq:   13
   9. Cost: -30088.30, Freq:   13
  10. Cost: -30088.54, Freq:   13

Classical benchmark...

=== Hardware-Optimized Performance Summary ===
Backend: lightning.gpu
Circuit gates: 120 (optimized)
Memory usage: 0.51 MB

Timing:
  Training: 1713.42s
  Sampling: 0.70s
  Classical: 0.16s
  Total quantum: 1714.11s

Results:
  Best quantum cost: -30088.5432
  Best classical cost: -30092.5023
  Quantum advantage: 1.000x

System Performance:
  Peak CPU: 0.0%
  Peak Memory: 6588.5 MB
  Duration: 1720.8s

Optimization Stats:
  Cache: 500/500 entries, 0.0% hit rate
  Best quantum cost: -30088.5432

Quantum Solution Analysis:
  Selected 5 bonds (target: 5):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US06051GMA49: Return=0.062, Risk=0.070
    4. US46647PDR47: Return=0.061, Risk=0.066
    5. US95000U3F88: Return=0.062, Risk=0.067
  Portfolio metrics:
    Cash flow: 0.0555 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2254 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.067 (6.7%)
    Total constraint violation: 0.1746

Classical Solution Analysis:
  Selected 6 bonds (target: 5):
    1. US00287YBX67: Return=0.054, Risk=0.045
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US46647PDR47: Return=0.061, Risk=0.066
    4. US55903VBC63: Return=0.074, Risk=0.062
    5. US95000U3F88: Return=0.062, Risk=0.067
    6. US06051GLU12: Return=0.062, Risk=0.068
  Portfolio metrics:
    Cash flow: 0.0633 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2394 (target: [0.4, 0.8])
    Average return: 0.062 (6.2%)
    Average risk: 0.063 (6.3%)
    Total constraint violation: 1.1639

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Final Quantum Solution Analysis:
  Selected 5 bonds (target: 5):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US06051GMA49: Return=0.062, Risk=0.070
    4. US46647PDR47: Return=0.061, Risk=0.066
    5. US95000U3F88: Return=0.062, Risk=0.067
  Portfolio metrics:
    Cash flow: 0.0555 (target: [3.0%, 6.0%])
    Risk characteristic: 0.2254 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.067 (6.7%)
    Total constraint violation: 0.1746

=== Real Data Integration Impact ===
Data source: vanguard_real (VCIT)
Real portfolio: $2,153,890,208 market value
Authentic bonds: 15 from 15 total positions
Real correlations: Avg 0.303
Actual risk-return: Duration 5.9y, Spread 91bp

=== Hardware Optimization Impact ===
Backend: lightning.gpu (hardware-optimized)
Circuit gates: 120 (memory-optimized)
Caching: Cache: 500/500 entries, 0.0% hit rate
Monitoring: Peak 6589MB
Configuration: 15 layers, 5 restarts

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -10113.02, Frequency = 15
Solution 2: Cost = -28091.72, Frequency = 14
Solution 3: Cost = -18104.35, Frequency = 14
Solution 4: Cost = -18113.44, Frequency = 14
Solution 5: Cost = -24106.47, Frequency = 14
Solution 6: Cost = -18114.30, Frequency = 13
Solution 7: Cost = -28098.57, Frequency = 13
Solution 8: Cost = -24106.78, Frequency = 13
Solution 9: Cost = -124.98, Frequency = 13
Solution 10: Cost = -18110.81, Frequency = 13

=== Creating Real Vanguard Portfolio Analysis Plots ===
Data: VCIT with 15 real bonds
Saving plots to: vanguard_quantum_analysis-2/
  Saved: vanguard_quantum_analysis-2/20250806_155724_performance_dashboard.png
  Saved: vanguard_quantum_analysis-2/20250806_155725_top_solutions_analysis.png
  Saved: vanguard_quantum_analysis-2/20250806_155726_portfolio_composition.png
  Saved: vanguard_quantum_analysis-2/20250806_155727_constraint_satisfaction.png
  Saved: vanguard_quantum_analysis-2/20250806_155728_system_performance.png
  Saved: vanguard_quantum_analysis-2/20250806_155729_qubo_structure.png
  Saved: vanguard_quantum_analysis-2/20250806_155731_solution_quality.png
```

## Real Data Example 5 output for exploring part 3 of steps1-5.py (With Lightning.GPU (24 qubits), QAOA Layers of 2)

```text
=== Hardware-Optimized Quantum Portfolio Optimization ===

=== Loading Real Vanguard Portfolio Data ===
Loading Vanguard VCIT bond portfolio data...
Successfully loaded 2629 bond positions from Vanguard portfolio
Dataset dimensions: 2629 assets x 278 features
Filtered to 2618 bond positions
Selected top 24 holdings by market value for optimization
  Returns calculated from OAS (credit spreads): 90 bps average
  Risk calculated from duration (avg: 5.91 years) and credit spreads
  Correlation matrix estimated from sector/credit clustering (avg: 0.318)
  Weights calculated from market values (largest: 0.117)

Portfolio Analysis Summary:
  Fund: VCIT ($3,047,237,649)
  Average Duration: 5.91 years
  Average Credit Spread: 90 basis points
  Expected Returns: [0.049, 0.074]
  Risk Measures: [0.044, 0.070]
  Sector Distribution: ['Financial', 'Industrial', 'Treasury Bond Portfolio']

=== Real Portfolio Characteristics ===
Fund: VCIT
Portfolio size: 24 bonds (memory-optimized)
Memory requirement: 0.25 GB
Note: For 31 qubits, you would need 32.0 GB RAM
Average duration: 5.91 years
Average credit spread: 90 basis points
Returns range: [0.049, 0.074]
Risk range: [0.044, 0.070]
Sample bonds: ['US91282CLN91', 'US87264ABF12', 'US06051GLH01']

=== Quantum Optimization Problem Setup ===
Real Vanguard bonds: 24, Target portfolio: 8
Data source: vanguard_real
Fund: VCIT ($3,047,237,649)

Real Market Parameters:
  Expected returns (m): [4.91, 7.35]%
  Return bounds (M): [5.89, 8.82]%
  Risk measures (i_c): [0.444, 0.699]
  Position weights (x_c): [1.478, 5.627]

Portfolio Constraints:
  Target basket size: 8 bonds
  Yield range: [3.0%, 6.0%]
  Duration target: 0.059
  Risk aversion: 2.0

Sample Real Bonds:
  US91282CLN91: Return=0.049, Risk=0.045
  US87264ABF12: Return=0.057, Risk=0.048
  US06051GLH01: Return=0.061, Risk=0.065
  US00287YBX67: Return=0.054, Risk=0.045
  US716973AE24: Return=0.059, Risk=0.068

Optimization ready: 24 real Vanguard bonds → 8 quantum portfolio

=== Building QUBO with Real Bond Data ===
Covariance matrix built from real bond correlations: (24, 24)
QUBO matrix Q shape: (24, 24)

=== Initializing Optimized Components ===
Hardware Config for 24 qubits:
  CPUs: 16, Memory: 7.3GB
  State memory needed: 0.25GB
  Recommended: {'qaoa_layers': 2, 'shots': 14672, 'restarts': 6, 'processes': 2, 'batch_size': 100, 'state_memory_gb': 0.25}
Memory Analysis for 24 qubits:
  Required: 0.25 GB
  Available: 7.34 GB
Using GPU acceleration (Lightning GPU)
Selected backend: lightning.gpu for 24 qubits
Circuit Analysis:
  Significant gates: 300
  Estimated memory: 256.02 MB

=== Starting Memory-Aware Training ===
Problem size: 24 qubits
QAOA layers: 2, Restarts: 6
Expected memory: 0.25 GB
Initial memory: 653.5 MB, Available: 5.81 GB

--- Restart 1/6 ---
  Iter   0: 2662.2465 | Cache: 1/500 entries, 0.0% hit rate
  Iter  50: 1871.0666 | Cache: 51/500 entries, 0.0% hit rate
  Iter 100: 744.2100 | Cache: 101/500 entries, 0.0% hit rate
  Iter 150:  61.8382 | Cache: 151/500 entries, 0.0% hit rate
New best: 283.7757

--- Restart 2/6 ---
  Iter   0: 1454.8002 | Cache: 201/500 entries, 0.0% hit rate
  Iter  50: -94.2396 | Cache: 251/500 entries, 0.0% hit rate
  Iter 100: -616.9449 | Cache: 301/500 entries, 0.0% hit rate
  Iter 150: 1466.1301 | Cache: 351/500 entries, 0.0% hit rate
New best: -849.9316

--- Restart 3/6 ---
  Iter   0: -886.3320 | Cache: 401/500 entries, 0.0% hit rate
  Iter  50: -114.1793 | Cache: 451/500 entries, 0.0% hit rate
  Iter 100: -352.3910 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -578.7280 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 4/6 ---
  Iter   0: 3335.9176 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50:  17.4718 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100:  65.5230 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: 127.4739 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 5/6 ---
  Iter   0: 177.5902 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -579.7770 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -306.7492 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: -672.4514 | Cache: 500/500 entries, 0.0% hit rate

--- Restart 6/6 ---
  Iter   0: -230.0905 | Cache: 500/500 entries, 0.0% hit rate
  Iter  50: -672.4463 | Cache: 500/500 entries, 0.0% hit rate
  Iter 100: -1088.8363 | Cache: 500/500 entries, 0.0% hit rate
  Iter 150: 140.4235 | Cache: 500/500 entries, 0.0% hit rate

=== Optimized Sampling ===
Memory Analysis for 24 qubits:
  Required: 0.25 GB
  Available: 7.34 GB
Using GPU acceleration (Lightning GPU)
Sampling: 14672 shots in 3.27s

Top quantum solutions:
   1. Cost: -72114.44, Freq:    6
   2. Cost: -72116.75, Freq:    4
   3. Cost: -72108.89, Freq:    4
   4. Cost: -72117.12, Freq:    4
   5. Cost: -72108.98, Freq:    3
   6. Cost: -72116.13, Freq:    3
   7. Cost: -72115.29, Freq:    3
   8. Cost: -72116.22, Freq:    3
   9. Cost: -72114.58, Freq:    3
  10. Cost: -72105.34, Freq:    3

Classical benchmark...

=== Hardware-Optimized Performance Summary ===
Backend: lightning.gpu
Circuit gates: 300 (optimized)
Memory usage: 256.02 MB

Timing:
  Training: 23734.30s
  Sampling: 3.27s
  Classical: 0.15s
  Total quantum: 23737.57s

Results:
  Best quantum cost: -72117.1201
  Best classical cost: -72122.6228
  Quantum advantage: 1.000x

System Performance:
  Peak CPU: 0.0%
  Peak Memory: 3590.4 MB
  Duration: 23739.5s

Optimization Stats:
  Cache: 500/500 entries, 0.0% hit rate
  Best quantum cost: -72117.1201

Quantum Solution Analysis:
  Selected 8 bonds (target: 8):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US06051GMA49: Return=0.062, Risk=0.070
    4. US031162DR88: Return=0.060, Risk=0.066
    5. US06051GLU12: Return=0.062, Risk=0.068
    6. US17327CAR43: Return=0.064, Risk=0.064
    7. US95000U3D31: Return=0.062, Risk=0.066
    8. US06051GKQ19: Return=0.060, Risk=0.060
  Portfolio metrics:
    Cash flow: 0.0900 (target: [3.0%, 6.0%])
    Risk characteristic: 0.3478 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.066 (6.6%)
    Total constraint violation: 0.0822

Classical Solution Analysis:
  Selected 9 bonds (target: 8):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US46647PDH64: Return=0.059, Risk=0.062
    3. US46647PDR47: Return=0.061, Risk=0.066
    4. US95000U3F88: Return=0.062, Risk=0.067
    5. US06051GLU12: Return=0.062, Risk=0.068
    6. US95000U3D31: Return=0.062, Risk=0.066
    7. US46647PDK93: Return=0.061, Risk=0.062
    8. US92343VGN82: Return=0.060, Risk=0.066
    9. US61747YEY77: Return=0.060, Risk=0.062
  Portfolio metrics:
    Cash flow: 0.0955 (target: [3.0%, 6.0%])
    Risk characteristic: 0.3793 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.065 (6.5%)
    Total constraint violation: 1.0562

=== Final Assessment ===
QUANTUM ADVANTAGE: Better constraint satisfaction

Final Quantum Solution Analysis:
  Selected 8 bonds (target: 8):
    1. US06051GLH01: Return=0.061, Risk=0.065
    2. US716973AE24: Return=0.059, Risk=0.068
    3. US06051GMA49: Return=0.062, Risk=0.070
    4. US031162DR88: Return=0.060, Risk=0.066
    5. US06051GLU12: Return=0.062, Risk=0.068
    6. US17327CAR43: Return=0.064, Risk=0.064
    7. US95000U3D31: Return=0.062, Risk=0.066
    8. US06051GKQ19: Return=0.060, Risk=0.060
  Portfolio metrics:
    Cash flow: 0.0900 (target: [3.0%, 6.0%])
    Risk characteristic: 0.3478 (target: [0.4, 0.8])
    Average return: 0.061 (6.1%)
    Average risk: 0.066 (6.6%)
    Total constraint violation: 0.0822

=== Real Data Integration Impact ===
Data source: vanguard_real (VCIT)
Real portfolio: $3,047,237,649 market value
Authentic bonds: 24 from 24 total positions
Real correlations: Avg 0.318
Actual risk-return: Duration 5.9y, Spread 90bp

=== Hardware Optimization Impact ===
Backend: lightning.gpu (hardware-optimized)
Circuit gates: 300 (memory-optimized)
Caching: Cache: 500/500 entries, 0.0% hit rate
Monitoring: Peak 3590MB
Configuration: 2 layers, 6 restarts

Top 10 Quantum Solutions Data for Plotting:
Solution 1: Cost = -60122.00, Frequency = 6
Solution 2: Cost = -60123.01, Frequency = 4
Solution 3: Cost = -66116.15, Frequency = 4
Solution 4: Cost = -52128.57, Frequency = 4
Solution 5: Cost = -70115.93, Frequency = 3
Solution 6: Cost = -70121.98, Frequency = 3
Solution 7: Cost = -52128.19, Frequency = 3
Solution 8: Cost = -30126.30, Frequency = 3
Solution 9: Cost = -60127.51, Frequency = 3
Solution 10: Cost = -70110.51, Frequency = 3

=== Creating Real Vanguard Portfolio Analysis Plots ===
Data: VCIT with 24 real bonds
Saving plots to: vanguard_quantum_analysis-2/
  Saved: vanguard_quantum_analysis-2/20250807_035343_performance_dashboard.png
  Saved: vanguard_quantum_analysis-2/20250807_035344_top_solutions_analysis.png
  Saved: vanguard_quantum_analysis-2/20250807_035345_portfolio_composition.png
  Saved: vanguard_quantum_analysis-2/20250807_035346_constraint_satisfaction.png
  Saved: vanguard_quantum_analysis-2/20250807_035347_system_performance.png
  Saved: vanguard_quantum_analysis-2/20250807_035348_qubo_structure.png
  Saved: vanguard_quantum_analysis-2/20250807_035350_solution_quality.png
```

## Key Features

### Adaptive Penalty System

- Dynamic constraint weight adjustment during optimization
- Automatic penalty scaling based on violation severity
- Improved convergence to feasible solutions

### QAOA Implementation

- Quantum Approximate Optimization Algorithm with 3 layers with lightning.gpu and shots are 36681 with 6 restarts, 4 processes, and batch size of 100
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
- Includes plots under the current_progress/vanguard_quantum_analysis and current_progress/vanguard_quantum_analysis-1

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
