# Comprehensive DRP Benchmark: Evaluating Deep Learning Models for Drug Response Prediction

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Abstract
This repository provides a standardized and rigorous benchmarking framework for evaluating state-of-the-art Deep Learning models in **Drug Response Prediction (DRP)**. 

Our goal is to move beyond simple performance metrics and strictly assess the robustness, scalability, and real-world generalizability of various DRP models. To achieve this, we implement unified data processing pipelines and multiple rigorous split scenarios, ensuring a fair and challenging evaluation environment for all models.

## 📦 Models Evaluated
- **[CSG2A]** Cell-Specific Gene-to-Attribute Deep Learning Framework
- **[DeepTTC]** Deep Transfer Learning for Drug Target Interaction
- *(More models will be added during the benchmarking process)*

---

## ⚙️ Installation

Clone the repository and set up the environment:

```bash
# 1. Clone the repository
git clone [https://github.com/YourUsername/DRP-Benchmark.git](https://github.com/YourUsername/DRP-Benchmark.git)
cd DRP-Benchmark

# 2. Create a virtual environment
conda create -n drp_benchmark python=3.8
conda activate drp_benchmark

# 3. Install dependencies
pip install -r requirements.txt
