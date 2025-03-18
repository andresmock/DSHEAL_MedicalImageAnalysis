# DSHEAL_project_FS25_GaMo
Medical Image Analysis - Malaria Detection Using Blood Smear Images

# Task
take a look at: https://github.zhaw.ch/ADLS-Digital-Health/DSHEAL-FS25/blob/main/project/assignment.md

# Structured project directory
```
├── data/                  # Sample data or processing scripts (NOT the entire dataset*)
├── analysis/              # Scripts/Jupyter Notebooks for analysis
├── src/                   # Main scripts for preprocessing, training, evaluation
├── models/                # Saved model weights and architectures
├── report/                # Final report and additional documentation
├── requirements.txt       # List of dependencies
├── README.md              # Instructions for running the project
├── results/               # Logs, confusion matrices, performance charts
└── LICENSE                # License file
```


# Environment Managment
## 1. Setting up the Environment
To create the Conda environment from the env.yaml file, run the following command:
```
conda env create --file env.yaml
conda activate DSHEAL_proj1_GaMo
```
## 2. Adding a New Package
Open the env.yaml file and add the new package under the dependencies section. \
Update the environment without recreating it:
```
conda env update --file env.yaml --prune
```

