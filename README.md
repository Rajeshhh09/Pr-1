# Deep Learning Practical — Project 1: Breast Cancer Classification

## 📋 Project Overview
This project implements and evaluates neural network classifiers ranging from a Single-Layer Perceptron (SLP) baseline to a fully regularised Multi-Layer Perceptron (MLP) using TensorFlow/Keras on the **Breast Cancer Wisconsin (Diagnostic)** dataset.

* **Dataset Source:** Scikit-Learn (`sklearn.datasets.load_breast_cancer`) — 569 samples, 30 numerical features, binary classification (Malignant vs Benign).
* **Tools Used:** TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.
* **Video Link:** [Project Walkthrough & Code Explanation Video]

---

## 🚀 7 Topics Covered

1. **Dataset Loading & Preprocessing:** Loaded features and targets, verified class distributions, checked feature correlations, and standardised features using `StandardScaler`.
2. **Single-Layer Perceptron (SLP) Baseline:** Built a single-neuron model (`binary_crossentropy`, 50 epochs) establishing baseline performance and confusion matrix metrics.
3. **Multi-Layer Perceptron (MLP) & Activations:** Constructed multi-layer neural networks and compared non-linear activation functions (`ReLU` vs `Tanh` vs `Sigmoid`).
4. **Early Stopping:** Implemented validation loss monitoring with `patience=15` and `restore_best_weights=True` to prevent overfitting.
5. **Dropout Regularization:** Integrated dropout layers (`rate=0.3`) to prevent neuron co-adaptation and improve generalisation.
6. **Weight Regularization (L1, L2, ElasticNet):** Applied L1, L2, and combined penalties to control coefficient magnitudes.
7. **Production Model & Final Comparison:** Consolidated dropout, L2 regularisation, and early stopping into an optimal model architecture.

---

## 📊 Plots & Visualizations

| Plot Name | Description & Caption |
| :--- | :--- |
| **Training Curves** | `plots/training_curves.png` — Training and validation loss trajectory showing steady convergence without overfitting. |
| **Activation Comparison** | `plots/activation_comparison.png` — Comparison of test accuracy across ReLU, Tanh, and Sigmoid activations. |
| **Dropout Rate Comparison** | `plots/dropout_comparison.png` — Evaluating drop rates from `0.0` to `0.5`; optimal performance achieved at `p=0.3`. |
| **Regularization Comparison** | `plots/regularization_comparison.png` — Performance comparison across unregularized, L1, L2, and ElasticNet models. |
| **Results Bar Chart** | `plots/results_bar_chart.png` — Comprehensive accuracy bar chart across all model variations. |

---

## 📈 Full Results Table

| Model | Architecture | Regularization | Dropout Rate | Early Stopping | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SLP Baseline** | 1 (Dense) | None | None | No | 0.9649 | 0.9583 | 0.9859 | 0.9718 |
| **MLP + Early Stopping** | 128 → 64 → 1 | None | None | Yes (p=15) | 0.9737 | 0.9718 | 0.9859 | 0.9787 |
| **MLP + Dropout (0.3)** | 128 → 64 → 1 | None | 0.3 | Yes (p=20) | 0.9825 | 0.9722 | 1.0000 | 0.9859 |
| **MLP + L2 (0.001)** | 128 → 64 → 1 | L2 (0.001) | None | Yes (p=20) | 0.9737 | 0.9718 | 0.9859 | 0.9787 |
| **Final Combined Model** | 128 → 64 → 1 | L2 (0.001) | 0.3 | Yes (p=20) | **0.9912** | **0.9861** | **1.0000** | **0.9930** |

---

## 💻 Commit History
* `Add SLP baseline — binary_crossentropy, 50 epochs, confusion matrix`
* `Build MLP — ReLU vs Tanh vs Sigmoid activation comparison`
* `Add Early Stopping — patience=15, restore_best_weights, stopping epoch plot`
