## NanoFilm_Geometry_Optimization

### Introduction

The goal of this project is to leverage a machine learning model trained on data simulated using COMSOL MULTIPHYSICS, provided by Edgar Briones Hernández, a researcher at ITESO. Our objective is to identify the optimal thickness combination of the Ta2O5, Al2O3, and SiO2 layers to minimize reflectance around a 600 nm wavelength.

### Discussion of Results

Our Support Vector Machine (SVM) regression model has demonstrated exceptional performance. It achieved an R² value of approximately 0.9997 on the test data, indicating an outstanding fit, and a Mean Squared Error (MSE) of around 0.0184, highlighting its remarkable accuracy.

To contextualize these results, we generated a dataset with all possible combinations of layer thicknesses and fed it into the model. The results, visualized in the accompanying graphs, indicate that combinations of thicknesses resulting in minimized reflectance around 600 nm typically involve Al2O3 at 100 nm and SiO2 at 40 nm, with other effective combinations being (110,20) and (110,30) respectively.

This comprehensive analysis confirms that the SVM model is the optimal choice for this specific problem, exhibiting both exceptional fit and precision.

### Repository Structure

```
NanoFilm_Geometry_Optimization/
│   README.md
│
├───Analysis
│       Analysis.ipynb
│       Find_maximizers.ipynb
│       Statistical_A.ipynb
│
├───Bayesian_Optimization
│   │   Optimization_analysis.ipynb
│   │   Optimization_svc.ipynb
│   │   Utils.py
│   │
│   └───Logs
│           svr.jsonl
│
├───Data
│       datos edgar.xlsx
│       prueba.csv
│       prueba2.csv
│       X_nueva_predicha
│       X_nueva_predicha_035
│
└───Models
        Datos_prueba.py
        model_svr.pkl
        pipeline_SVM.py
        standard_scaler.pkl
```

This repository contains all the necessary resources and analyses for optimizing the geometry of nanofilms to minimize reflectance. It includes:

- **Analysis**: Jupyter notebooks for statistical analysis and identifying optimal thickness combinations.
- **Bayesian_Optimization**: Notebooks and utilities for Bayesian optimization, including logs.
- **Data**: Datasets used for training and validation.
- **Models**: Python scripts and serialized models for SVM and data preprocessing.

Explore the notebooks and scripts to understand the methodology and results of our optimization process.