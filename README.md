# **DSHEAL_project_FS25_GaMo**
Medical Image Analysis - Malaria Detection Using Blood Smear Images

## Table of Content
- [Task](#task)
- [Short Project Description (Abstract)](#short-project-description-abstract)
- [Structured Project Directory](#structured-project-directory)
- [Dataset Details (Source & Statistics)](#dataset-details-source--statistics)
- [Data Preprocessing](#data-preprocessing)
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
├── env.yaml                    # Conda environment definition file
├── LICENSE                     # License file
└── README.md                   # Project overview and usage instructions
```

---

## Dataset Details (Source & Statistics)

The dataset used in this project is publicly available on Kaggle:  
https://www.kaggle.com/datasets/maestroalert/malaria-split-dataset/data

It contains cell images labeled as either **Parasitized** or **Uninfected**, already split into training, validation, and test sets.

### File Type Summary
- Total image files: **27,558**
- Format: `.png` only

### Number of Images per Class
| Set        | Parasitized | Uninfected | Total  |
|------------|-------------|------------|--------|
| Train      | 11,024      | 11,024     | 22,048 |
| Validation | 1,378       | 1,377      | 2,755  |
| Test       | 1,377       | 1,378      | 2,755  |
| **Total**  | **13,779**  | **13,779** | **27,558** |

### Image Dimension Analysis
- Images vary in size, with most clustered around **128×128 pixels**.
- Resizing to **128×128** was chosen for model input consistency.
- (See plots in `analysis/data_expd.ipynb` for more details.)

### RGB Mean & Std (All Sets Combined)
Calculated across all train, validation, and test images:

- **Mean:** `[0.5295, 0.4239, 0.4530]`
- **Std:** `[0.3323, 0.2682, 0.2821]`

These values were used to normalize images during preprocessing.

---

## Data Preprocessing

Before feeding the images into the neural network, a set of preprocessing and transformation steps is applied to ensure consistent input dimensions and improve model generalization.

### Common Preprocessing Steps (All Sets)
- **Resize:** All images are resized to **128×128 pixels** to standardize input dimensions.
- **Normalization:** Images are normalized using the RGB mean and standard deviation computed over the entire dataset:
  - `mean = [0.529, 0.424, 0.453]`
  - `std = [0.332, 0.268, 0.282]`
  - This scales pixel values to approximately the range [-1, 1].

### Data Augmentation (Train Set Only)
To improve generalization and prevent overfitting, several augmentation strategies can be applied to the training set via the `aug_level` parameter:

| Augmentation Level | Transformations Applied                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------|
| `none`             | Resize → ToTensor → Normalize                                                                 |
| `basic` *(default)*| Resize → RandomHorizontalFlip → RandomRotation(+-15°) → ToTensor → Normalize                  |
| `strong`           | Resize → RandomHorizontalFlip/VerticalFlip → RandomRotation(+-180°) → ColorJitter → Normalize |

These options can be set via the `get_train_transform()` function.

### Dataloader Setup
- Images are loaded using `torchvision.datasets.ImageFolder()`, which automatically assigns:
  - `0 = Parasitized` (based on folder name)
  - `1 = Uninfected`
- Data is then wrapped into PyTorch `DataLoader` objects with batching and shuffling enabled (for training).
- Validation and test sets are never augmented and use only resizing and normalization.

### Data Path Handling
The dataset root path is stored in a separate `data_path.txt` file (not tracked by Git).  
This path is automatically loaded via a utility function to keep code clean and portable.

---

> 📌 Example:  
> You can create all data loaders by calling:  
> `train_loader, val_loader, test_loader = create_dataloaders(aug_level="basic", batch_size=32)`


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



