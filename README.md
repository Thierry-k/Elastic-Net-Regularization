## Elastic Net Performance Analysis

This project explores the performance of the **Elastic Net regularization method** in high-dimensional linear regression models, particularly when features are correlated. It was developed as part of the coursework for **MAT4376 – High Dimensional Data Analysis** at the University of Ottawa.

## 📌 Objective

To assess how Elastic Net performs compared to its constituent methods (Lasso and Ridge) by:
- Simulating datasets with varying feature correlations.
- Tuning the Elastic Net mixing parameter (**α**) and penalty term (**λ**) using cross-validation.
- Evaluating predictive accuracy and model sparsity.
- Conducting a Monte Carlo simulation across multiple correlation scenarios.

## 🛠 Features

- **Simulation of Multivariate Normal Data** using Cholesky decomposition
- **Elastic Net Regression** for multiple values of α (0.1, 0.5, 0.9)
- **Cross-validation** to optimize λ
- **MSE-based performance comparison** across α values
- **Monte Carlo simulation** to evaluate model stability across ρ ∈ [−0.9, 0.9]
- **Visualization** of MSE trends vs. correlation strength

## 📁 Project Structure

```
├── elastic_net_project_code.md  # Complete R markdown for simulation, modeling, and plotting
├── README.md                    # Project description and usage instructions
├── figure-gfm/                  # Output graphs
└── report/                      # Final report
```

## 📈 Sample Output

- Tables of MSEs for different α values
- MSE vs. ρ plots demonstrating optimal α under various correlation structures

## 📦 Requirements

This project was developed in **R**. Required packages:
```r
install.packages(c("MASS", "glmnet", "ggplot2", "reshape2"))
```

## 🚀 How to Run

1. Clone the repository
2. Open `elastic_net_project_code.md` in RStudio or another R environment
3. Run the full script to simulate data, fit models, and generate outputs

## 📄 License

This project is for academic use only as part of university coursework. No commercial license granted.

## 👤 Author

Thierry Kubwimana  
University of Ottawa | MAT4376 (Winter 2024)
