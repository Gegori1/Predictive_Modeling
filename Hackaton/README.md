# Hackaton-Predictive Modeling

This repository contains the code to analyze and model some data.

The goal is to explore, clean, and analyze the data before the modeling process, using statistical tools. The usage of the models is restricted to linear models.

## Repository Structure

```
│   analysis.ipynb
│   danalysis.ipynb
│   pipeline.py
│   README.md
│   Unknown_dataset.csv
│   
└───data
        Audit_test_.csv
        Audit_train_.csv
        Audit_unknown_.csv
```

- **analysis.ipynb**: Contains the results of the initial data exploration process.
- **danalysis.ipynb**: Contains additional data exploration and analysis results.
- **pipeline.py**: Uses the identified feature engineering methods to clean the data and forecast a new model.
- **Unknown_dataset.csv**: The dataset used for analysis and modeling.
- **data/**: Directory containing the training, testing, and unknown datasets for the audit.

## Usage

Run the notebooks to explore and analyze the data. Use `pipeline.py` to apply the found feature engineering methods and build a new linear model.