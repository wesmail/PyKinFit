# Kinematic Fitting Algorithm in Python

## Introduction

Kinematic fitting is a powerful tool used in the field of particle physics to improve the precision of measured quantities and to reduce background noise. It is particularly useful in the exclusive analysis of particle reactions. The algorithm aims to estimate the true values of track parameters as close as possible to the measured values while fulfilling a set of constraints given by the kinematics of the reaction under study.

### Mathematical Details

The core of the kinematic fitting algorithm is the minimization of the \(\chi^{2}\) function, which is defined as:

\[
\chi^2 = (y - \eta)^T V^{-1} (y - \eta)
\]

Here, \(y\) represents the vector of measured values, \(\eta\) represents the vector of true values, and \(V\) is the covariance matrix of the measured values. The minimization is performed iteratively.

#### Lagrange Multipliers

The algorithm employs Lagrange multipliers to handle constraints. The Lagrange function \(L\) is formulated as:

\[
\chi^2 = (y - \eta)^T V^{-1} (y - \eta) + 2\lambda^T f(\eta, \xi)
\]

Here, \(f(\eta, \xi)\) represents the constraint equations, and \(\lambda\) are the Lagrange multipliers.

#### Jacobian Matrices

The Jacobian matrices \(F_{\eta}\) and \(F_{\xi}\) are crucial for the algorithm. They are partial derivatives of the constraint equations with respect to \(\eta\) and \(\xi\), respectively. These matrices are calculated using PyTorch's automatic differentiation engine.

\[
F_{\eta} = \frac{\partial f}{\partial \eta}, \quad F_{\xi} = \frac{\partial f}{\partial \xi}
\]

#### Iterative Minimization

The algorithm iteratively updates the parameters \(\eta\) and \(\xi\) to minimize \(\chi^2\). In each iteration, new approximations for \(\eta\) and \(\xi\) are found, and the \(\chi^2\) value is updated accordingly.

## Features

- **PyTorch Integration**: This implementation utilizes PyTorch's automatic differentiation tool to find the Jacobian matrices \(F_{\eta}\) and \(F_{\xi}\).
  
- **Generic Implementation**: The algorithm is designed to be generic. Users can provide "measured" and "unmeasured" parameters in any representation (track parameterization) along with the covariance matrix of the measured parameters and the constraint equation(s).

## How It Works

1. **Input Parameters**: The user provides the measured and unmeasured parameters, the covariance matrix of the measured parameters, and the constraint equations.

2. **Iterative Minimization**: The algorithm minimizes the \(\chi^{2}\) function iteratively to find the best estimates of the true observables.

3. **Output**: The algorithm outputs an improved set of parameters that fulfill the kinematic constraints.

## Usage

[Provide code examples and usage instructions here]

## Installation

[Provide installation instructions here]

## License

[Provide license information here]
