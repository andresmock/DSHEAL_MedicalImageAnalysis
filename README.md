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



