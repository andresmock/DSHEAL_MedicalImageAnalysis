# **DSHEAL_project_FS25_GaMo**
Medical Image Analysis - Malaria Detection Using Blood Smear Images

## Table of Content
- [Task](#task)
- [Short Project Description (Abstract)](#short-project-description-abstract)
- [Structured Project Directory](#structured-project-directory)
- [Dataset Details (Source, Preprocessing)](#dataset-details-source-preprocessing)
- [Installation and Usage Instructions](#installation-and-usage-instructions)
  - [Data Download](#data-download)
  - [Environment Management: Setting up the Environment](#environment-management-setting-up-the-environment)
- [Evaluation](#evaluation)
- [Model Performance Summary](#model-performance-summary)
- [Team Members and Contributions](#team-members-and-contributions)

## Task
take a look at: https://github.zhaw.ch/ADLS-Digital-Health/DSHEAL-FS25/blob/main/project/assignment.md



## Short project description (abstract)


### Structured project directory
```
├── data/                       # Sample data or processing scripts (NOT the entire dataset)
│   ├── data_loader.py              # Data loading and transformation
│   ├── malaria_optuna.db           # Optuna study database
├── analysis/                   # Jupyter notebooks and scripts for analysis
│   ├── data_expd.ipynb             # Exploratory data analysis
│   ├── hyperparam.ipynb            # Hyperparameter analysis
│   ├── LIME_explanation.py         # LIME exploration script
│   ├── threshold_analysis.ipynb    # Threshold decision analysis
├── src/                        # Main scripts for training and evaluation
│   ├── evaluate.py                 # Evaluate the best model
│   ├── hyperparam_search.py        # Hyperparameter tuning with Optuna
│   ├── test_model.py               # Model testing script
│   ├── train_model_cv.py           # Model training with K-Fold cross-validation
│   ├── train_model.py              # Model training with predefined split
├── models/                     # Saved model weights and architectures
│   ├── best_malaria_model_*        # Best model from each run
│   ├── malaria_model_1.py          # Base model
│   ├── malaria_model_flex.py       # Flexible model, configurable via Optuna
├── report/                     # Final report and related documentation
│   ├── analysis_plots/             # Plots from exploratory and performance analysis
│   ├── performance_plots/          # Visualizations from Weights & Biases
├── requirements.txt            # List of Python dependencies
├── README.md                   # Instructions for using the project
├── results/                    # Evaluation results
│   ├── evaluation_best_*           # Evaluation outputs per model
├── wandb/                      # Weights & Biases log directory
│   ├── run-*                       # Logs for each training run
└── LICENSE                     # License file

```

## Dataset details (source, preprocessing)
source: https://www.kaggle.com/datasets/maestroalert/malaria-split-dataset/data


## Installation and usage instructions
### Data download
1. Download the dataset from: https://www.kaggle.com/datasets/maestroalert/malaria-split-dataset/data

After extraction, the directory structure should look like this:
```
├── your_data_path/data/ 
    ├── test
        ├── Parasitized
        ├── Uninfected
    ├── train
        ├── Parasitized
        ├── Uninfected
    ├── val
        ├── Parasitized
        ├── Uninfected
```

2. Replace `your_data_path` with the actual path to your data.
3. Create a file named `data_path.txt` and store your local data path in it.

### Environment Management: Setting up the Environment
To create and activate the Conda environment from the `env.yaml` file, run:
```
conda env create --file env.yaml
conda activate DSHEAL_proj1_GaMo
```

## Evaluation
To evaluate the best model (or alternatively test all models in the models/ directory), run:
```
python src\evaluate.py --data_path "your_data_path/data"
```
Replace "your_data_path/data" with the actual path to your dataset.

## Model performance summary

## Team members and contributions
**Mike Gasser**  
ZHAW School of Engineering  
gassemik@students.zhaw.ch  

**Andres Mock**  
ZHAW School of Engineering  
mockand1@students.zhaw.ch



