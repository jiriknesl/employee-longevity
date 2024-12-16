# Employee Retention Predictor

This is a Bayesian model that predicts employee retention and analyzes potential reasons for turnover based on various factors including workload, salary, and tenure.

## Features

- Predicts expected length of employment
- Calculates probability of employee leaving
- Analyzes most likely reason for departure (if applicable)
- Provides confidence intervals for predictions
- Uses Bayesian inference for robust uncertainty quantification

## Prerequisites

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The model requires two CSV files:
1. Historical data file for training (see `data.example.csv`)
2. Current employees file for prediction (see `current_employees.example.csv`)

### Running the Model

```bash
python run.py --learning data.csv --input current_employees.csv --output report.csv
```

### Input File Formats

#### Historical Data (learning) CSV
Must include the following columns:
- `employee_id`: Unique identifier (email or employee ID)
- `months_in_company`: Duration of employment
- `workload_quota`: Percentage of workload achievement (e.g., 0.95 = 95%)
- `salary_ratio`: Employee's salary as percentage of reference salary (e.g., 1.10 = 110%)
- `stayed`: Boolean (1 = still employed, 0 = left)
- `termination_reason`: Code for why employee left (0 = N/A, 1 = underperformance, 2 = better offer)

#### Current Employees CSV
Must include:
- `employee_id`: Unique identifier (email or employee ID)
- `months_in_company`: Current duration of employment
- `workload_quota`: Current workload achievement
- `salary_ratio`: Current salary ratio

### Output Report

The output CSV will contain all input columns plus:
- `employee_id`: Employee identifier from input
- `predicted_longevity`: Expected total months of employment
- `longevity_ci_50_low/high`: 50% confidence interval
- `longevity_ci_75_low/high`: 75% confidence interval
- `probability_of_leaving`: Chance of employee leaving (0-1)
- `predicted_reason`: Most likely reason if leaving
- `prob_stays`: Probability of staying
- `prob_underperformance`: Probability of leaving due to underperformance
- `prob_better_offer`: Probability of leaving for a better offer

## Example Files

The repository includes example files:
- `data.example.csv`: Example historical data
- `current_employees.example.csv`: Example current employees data

You can test the model using these files:

```bash
python run.py --learning data.example.csv --input current_employees.example.csv --output report.csv
```

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.