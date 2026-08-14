# BioGDR
The official implementation of "Multimodal interpretable deep learning for transcriptome-informed precision oncology and drug mechanism analysis".  
<img width="4049" height="4715" alt="figure1" src="https://github.com/user-attachments/assets/40c87c47-297d-4938-8f85-fad98b691103" />



## 1. Requirements
  - python=3.8
  - pytorch=1.11.0
  - torchvision
  - torchaudio
  - cudatoolkit=11.3
  - rdkit=2020.09.1
  - scikit-learn=1.0.2
  - dgl-cuda11.3=0.8.1
  - mkl=2024.0.0
  - dgllife==0.2.9

We recommend using the provided environment configuration file `env.yml`.  
You can create and activate the environment using mamba:
```
mamba env create -f env.yml
conda activate biogdr
```
We suggest using a V100 GPU with 32GB memory for training.  

## 2. Data Preparation
All preprocessed datasets and corresponding data splits are available at:  
https://zenodo.org/records/15718571  
All core experiments can be run directly using the provided preprocessed datasets.  
No raw data download or preprocessing is required.  


### 2.1 Download and Setup
1. Download the datasets from Zenodo:  
 - PRISM.zip  
 - GDSC.zip  

2. Unzip:
```
unzip PRISM.zip
unzip GDSC.zip
```
3. Place them under the `data/` directory:
```
data/
 ├── PRISM/
 ├── GDSC/
```

### 2.2 Dataset Structure

Each dataset contains:
```
PRISM/
 ├── cell/        # cell line features
 ├── drug/        # drug features
 ├── Experiment/  # experiment splits
      ├── random/
      ├── drug_blind/
      ├── ...

GDSC/
 ├── cell/        # cell line features
 ├── drug/        # drug features
 ├── Experiment/  # experiment splits
      ├── random/
```
Each split directory contains a `data_split/` folder with train/validation/test indices.  

### 2.3 Split definitions

The following data splits are provided:

 - **Random:**  
  Random split (8:1:1), repeated 5 times
 - **Unseen drug (drug_blind):**  
  Drugs in the test set are not present in the training or validation sets, using an 8:1:1 split, repeated 5 times
 - **Unseen cell (cell_blind):**  
  Cell lines in the test set are not present in the training or validation sets, using an 8:1:1 split, repeated 5 times
 - **Unseen dissimilar drug (drug_sim_blind):**  
  10-fold cross-validation based on drug similarity
 - **Unseen dissimilar cell (cell_sim_blind):**  
  10-fold cross-validation based on cell similarity

## 3. Run Experiments
Use the following scripts to run the experiments reported in the manuscript.  

**PRISM**
```
bash run_PRISM_random.sh
bash run_PRISM_cell_blind.sh
bash run_PRISM_cell_sim_blind.sh
bash run_PRISM_drug_blind.sh
bash run_PRISM_drug_sim_blind.sh
```
**GDSC**
```
bash run_GDSC_random.sh
```
For GDSC, only the random split is provided, consistent with the experiments reported in the manuscript.  

## 4. Example Usage (Custom Dataset)
Example datasets are provided for demonstration:
```
bash Example_train.sh
bash Example_eval.sh
```
These example scripts are provided to illustrate how to train and evaluate BioGDR on custom datasets.
 
