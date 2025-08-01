# **DSHEAL_project_FS25_GaMo**
Medical Image Analysis - Malaria Detection Using Blood Smear Images

## Table of Content
- [Task](#task)
- [Short Project Description (Abstract)](#short-project-description-abstract)
- [Structured Project Directory](#structured-project-directory)
- [Dataset Details (Source & Statistics)](#dataset-details-source--statistics)
- [Data Preprocessing](#data-preprocessing)
- [Model Performance Summary](#model-performance-summary)
- [Installation and Usage Instructions](#installation-and-usage-instructions)
- [Evaluation](#evaluation)
- [Team Members and Contributions](#team-members-and-contributions)


---


## Task
take a look at: https://github.zhaw.ch/ADLS-Digital-Health/DSHEAL-FS25/blob/main/project/assignment.md


---


## Short Project Description (Abstract)

Malaria is a life-threatening disease that continues to impact millions globally, particularly in regions with limited access to healthcare resources. Manual diagnosis through microscopic analysis of blood smear images is both time-consuming and heavily reliant on expert knowledge. In this project, we developed a convolutional neural network (CNN) to automate the classification of malaria-infected cells using a publicly available dataset. The dataset contained labeled images of parasitized and uninfected cells, which were preprocessed through resizing, normalization, and augmentation techniques to improve model generalization. Our baseline model achieved a classification accuracy of approximately 95\%, comparable to human expert performance.

To enhance model reliability, we explored hyperparameter tuning and interpretability methods, including threshold adjustment and LIME (Local Interpretable Model-agnostic Explanations). While threshold tuning successfully reduced false negatives, a critical factor in clinical settings, visual explanations from LIME revealed limited visible differences between correct and incorrect predictions, highlighting the ongoing challenges in model interpretability. Despite these limitations, our findings demonstrate that lightweight CNNs can serve as effective diagnostic support tools. Future work should focus on incorporating more diverse datasets, applying transfer learning, and validating the model in real-world clinical environments.

All code, trained models, and analysis results are included for full reproducibility.


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


## Model Performance Summary

**Evaluated Model:** `best_malaria_model_attmy8cd_epoch7`  
**Augmentation Level:** `basic`  
**Evaluation Set:** Test Set (2,755 images)

### Classification Report (Threshold = 0.50)

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Uninfected   | 0.9640    | 0.9339 | 0.9487   | 1377    |
| Parasitized  | 0.9360    | 0.9652 | 0.9503   | 1378    |
| **Accuracy** |           |        | **0.9495** | **2755** |
| **Macro avg**| 0.9500    | 0.9495 | 0.9495   | 2755    |
| **Weighted avg** | 0.9500 | 0.9495 | 0.9495   | 2755    |

**Confusion Matrix (Threshold = 0.50):**

- True Positives (TP, Parasitized correctly classified): 1330  
- True Negatives (TN, Uninfected correctly classified): 1286  
- False Positives (FP, Uninfected misclassified as Parasitized): 91  
- **False Negatives (FN, Parasitized missed): 48**  

---

### Classification Report (Adjusted Threshold = 0.35)

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Uninfected   | 0.9746    | 0.9187 | 0.9458   | 1377    |
| Parasitized  | 0.9231    | 0.9761 | 0.9489   | 1378    |
| **Accuracy** |           |        | **0.9474** | **2755** |
| **Macro avg**| 0.9489    | 0.9474 | 0.9473   | 2755    |
| **Weighted avg** | 0.9488 | 0.9474 | 0.9473   | 2755    |

**Confusion Matrix (Threshold = 0.35):**

- True Positives (TP): 1345  
- True Negatives (TN): 1265  
- False Positives (FP): 112  
- **False Negatives (FN): 33**  

---

### Interpretation

Adjusting the classification threshold from 0.50 to 0.35 significantly reduced the number of false negatives (from 48 to 33), which is crucial for medical diagnostics like malaria detection. Although this led to a slight increase in false positives (from 91 to 112), the overall accuracy remained high and nearly unchanged (~94.7%).

This trade-off aligns the model’s behavior with real-world priorities, where missing a malaria case is considered more critical than a false alarm.


---


## Installation and Usage Instructions

### 1. Clone the Repository

Clone this repository and navigate into the project folder:

```bash
git clone https://github.zhaw.ch/mockand1/DSHEAL_project_FS25_GaMo.git
cd DSHEAL_project_FS25_GaMo
```

### 2. Environment Setup (via Conda)

Create and activate the Conda environment using the `env.yaml` file:

```bash
conda env create --file env.yaml
conda activate DSHEAL_proj1_GaMo
```

### 3. Data Download and Setup

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/maestroalert/malaria-split-dataset/data

2. Extract the dataset so that the folder structure looks like this:

```
your_data_path/data/
├── train/
│   ├── Parasitized/
│   └── Uninfected/
├── val/
│   ├── Parasitized/
│   └── Uninfected/
└── test/
    ├── Parasitized/
    └── Uninfected/
```

> Make sure the folders `train`, `val`, and `test` are directly inside the `data/` directory.  
> You do **not** need to rename or rearrange the subfolders — they are already correctly labeled.

3. Create a file named `data_path.txt` in the **root of the cloned repository** (`DSHEAL_project_FS25_GaMo`) and insert your local absolute path to the `data/` directory.

Example content of `data_path.txt` (Windows):

```
C:\ZHAW_local\project_work\data
```


---


## Evaluation
To evaluate the best model (or alternatively test all models in the models/ directory), run:
```
python src\evaluate.py --data_path "your_data_path/data"
```
Replace "your_data_path/data" with the actual path to your dataset.


---


## Team members and contributions
**Mike Gasser**  
ZHAW School of Engineering  
gassemik@students.zhaw.ch  

**Andres Mock**  
ZHAW School of Engineering  
mockand1@students.zhaw.ch



