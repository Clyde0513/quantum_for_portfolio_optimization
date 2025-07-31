# Quantum Portfolio Optimization

This project implements a quantum approach to portfolio optimization using Variational Quantum Eigensolver (VQE) and PennyLane. The code is inspired by the methodology described in [The Wiser's Quantum Portfolio Optimization](https://www.thewiser.org/quantum-portfolio-optimization). This is also a Hackathon for Wiser.

## Overview

The goal of this project is to solve a Quadratic Unconstrained Binary Optimization (QUBO) problem for portfolio optimization. The problem is mapped to a quantum Hamiltonian, and the ground state of the Hamiltonian represents the optimal portfolio.

### Key Components

1. **Toy Problem Setup**:
   - Defines a toy dataset with 6 bonds.
   - Includes market parameters such as minimum/maximum trade, basket inventory, and minimum increment.
   - Sets global constraints like maximum bonds in a portfolio and residual cash flow bounds.

2. **QUBO Formulation**:
   - Constructs the QUBO matrix `Q` and vector `q` based on penalties for:
     - Main objective: Minimizing the deviation from the target portfolio.
     - Basket size constraint.
     - Residual cash flow bounds.
     - Characteristic bounds.
   - Symmetrizes the QUBO matrix.

3. **Mapping to Quantum Hamiltonian**:
   - Converts the QUBO problem into a Hamiltonian using PennyLane's `qml.Hamiltonian`.

4. **Variational Quantum Eigensolver (VQE)**:
   - Uses the `StronglyEntanglingLayers` ansatz with 4 layers.
   - Optimizes the parameters using `qml.AdamOptimizer`.

5. **Sampling and Interpretation**:
   - Samples the quantum circuit to obtain bit strings representing portfolios.
   - Converts samples to bit strings and identifies the most frequent portfolio.

## Current Status

- **Ansatz Layers**: Increased to 4 for better expressiveness.
- **Sampling Shots**: Increased to 5000 for improved accuracy.
- **Sampling Logic**: Sample by applying `qml.PauliZ` to each wire individually.

## Next Steps

1. **Refine Ansatz**:
   - Experiment with alternative ansatz designs to balance expressiveness and trainability.
   - Test with different numbers of layers and entanglement patterns.

2. **Optimize Parameters**:
   - Adjust learning rate and number of iterations for better convergence.
   - Explore other optimizers like `qml.GradientDescentOptimizer` or `qml.QNGOptimizer`.

3. **Expand Problem Scope**:
   - Increase the number of bonds in the dataset.
   - Introduce additional constraints or characteristics.

4. **Performance Analysis**:
   - Compare results with classical optimization methods.
   - Analyze the impact of quantum noise and hardware limitations.

5. **Documentation and Visualization**:
   - Add detailed comments and explanations in the code.
   - Visualize the optimization process and results using plots.

## How to Run

1. Install dependencies:
   ```bash
   pip install [the imports provided]
   ```

2. Run the script:
   ```bash
   python step1-3.py
   ```

3. Review the output:
   - The script prints the best portfolio (bit string) and its frequency.

## References

- [The Wiser's Quantum Portfolio Optimization](https://www.thewiser.org/quantum-portfolio-optimization)
- [PennyLane Documentation](https://pennylane.ai/)

---
