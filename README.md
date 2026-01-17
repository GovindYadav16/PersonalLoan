# Personal Loan ML Project - AllLife Bank

A machine learning project for predicting personal loan purchases for AllLife Bank customers using various classification algorithms.

## Business Context

**AllLife Bank** is a US bank with a growing customer base, primarily consisting of liability customers (depositors). The bank has a small number of borrowers (asset customers) and aims to rapidly expand its loan business to increase interest earnings. 

**Problem Statement:** Management wants to convert existing liability customers into personal loan customers while retaining them as depositors. A campaign last year for liability customers achieved a conversion rate of over 9%, encouraging the retail marketing department to develop more targeted campaigns.

**Objective:**
- To predict whether a liability customer will buy personal loans
- To understand which customer attributes are most significant in driving these purchases
- To identify which segments of customers to target more effectively

## Project Structure

```
personal-loan-ml/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned & feature-engineered data
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── data_ingestion.py       # Load & validate data
│   ├── preprocessing.py        # Feature engineering
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
│   ├── predict.py              # Inference
│   └── utils.py                # Utility functions
│
├── models/
│   └── model.pkl               # Saved trained model
│
├── config/
│   └── config.yaml             # Hyperparameters & paths
│
├── requirements.txt
├── README.md
├── main.py                     # Single entry point
└── setup.py                    # Optional package setup
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Train a model using the configuration file:
```bash
python main.py --mode train
```

### 2. Evaluation

Evaluate the trained model:
```bash
python main.py --mode evaluate
```

### 3. Prediction

Make predictions on new data:
```bash
# Using a sample file (without Personal_Loan column)
python main.py --mode predict --data data/raw/new_customers_sample.csv --output predictions.csv

# Or use the full dataset (Personal_Loan will be preserved in output for comparison)
python main.py --mode predict --data data/raw/loan_data.csv --output predictions.csv
```

**Note:** 
- The input data file should have the same structure as the training data (same columns except `Personal_Loan` can be omitted)
- The `ID` column will be automatically excluded from features
- If `Personal_Loan` column exists in the input, it will be included in the output for comparison
- A sample file `new_customers_sample.csv` (5 rows without Personal_Loan) is provided for testing

## Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Model hyperparameters
- Preprocessing strategies
- Evaluation metrics

## Dataset Description

The dataset contains 5000 customer records with the following features:

| Column | Description |
|--------|-------------|
| **ID** | Customer ID (unique identifier) |
| **Age** | Customer's age in completed years |
| **Experience** | Number of years of professional experience |
| **Income** | Annual income of the customer (in thousand dollars) |
| **ZIPCode** | Home Address ZIP code |
| **Family** | The family size of the customer |
| **CCAvg** | Average spending on credit cards per month (in thousand dollars) |
| **Education** | Education Level: 1=Undergrad, 2=Graduate, 3=Advanced/Professional |
| **Mortgage** | Value of house mortgage if any (in thousand dollars) |
| **Personal_Loan** | **Target variable**: Whether customer accepted the personal loan (1=Yes, 0=No) |
| **Securities_Account** | Has securities account with bank (1=Yes, 0=No) |
| **CD_Account** | Has certificate of deposit account (1=Yes, 0=No) |
| **Online** | Uses Internet banking facilities (1=Yes, 0=No) |
| **CreditCard** | Uses credit card issued by other bank (1=Yes, 0=No) |

## Data Requirements

- The dataset is already placed in `data/raw/loan_data.csv`
- The configuration file is set up with the correct target column (`Personal_Loan`)
- The ID column is automatically excluded from model features

## Modules

- **data_ingestion.py**: Loads and validates datasets
- **preprocessing.py**: Handles missing values, encoding, scaling
- **train.py**: Trains ML models (Random Forest, Logistic Regression)
- **evaluate.py**: Evaluates model performance with metrics
- **predict.py**: Makes predictions on new data
- **utils.py**: Helper functions for logging, file operations

## Notes

- Ensure your dataset is placed in `data/raw/` before training
- Update the `target_column` in `config.yaml` based on your dataset
- The model supports both Random Forest and Logistic Regression

## License

[Add your license here]
