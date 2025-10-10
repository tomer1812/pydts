[![pypi version](https://img.shields.io/pypi/v/pydts)](https://pypi.org/project/pydts/)
[![Tests](https://github.com/tomer1812/pydts/workflows/Tests/badge.svg)](https://github.com/tomer1812/pydts/actions?workflow=Tests)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://tomer1812.github.io/pydts)
[![codecov](https://codecov.io/gh/tomer1812/pydts/branch/main/graph/badge.svg)](https://codecov.io/gh/tomer1812/pydts)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16991039.svg)](https://doi.org/10.5281/zenodo.16991039)

# Discrete Time Survival Analysis  

*PyDTS* is a Python package designed for discrete-time survival analysis with competing risks, offering tools for model fitting, evaluation, and simulation.

For details, usage examples, and API information - check out the package 
[documentation](https://tomer1812.github.io/pydts/) 

*PyDTS* offers:

- Discrete-time competing-risks regression models.
- Automated procedures for hyperparameter tuning.
- Sure Independence Screening methods for feature selection.
- Model evaluation metrics for predictive accuracy and calibration.
- Simulation tools for generating synthetic datasets for research and testing.

Additional simulations and illustrative examples are available in [Meir and Gorfine (2025), Discrete-Time Competing-Risks Regression with or without Penalization, Biometrics (2025)](https://academic.oup.com/biometrics/article/81/2/ujaf040/8120014), and in the accompanying [Github Repository](https://github.com/tomer1812/DiscreteTimeSurvivalPenalization/tree/main)

## Installation

*PyDTS* can be installed using PyPI as follows:

```console
pip install pydts
```

## Dependencies

*PyDTS* supports Python versions 3.9–3.13.  

The package requires the following dependencies (with version constraints chosen for compatibility across Python and NumPy/SciPy releases):  

- *NumPy*  
  - Python 3.9: `>=1.26,<2.1`  
  - Python 3.10: `>=1.26,<2.3`  
  - Python 3.11–3.13: `>=1.26` (including NumPy 2.x)  

- *SciPy*  
  - Python 3.9: `>=1.13,<1.14`  
  - Python 3.10: `>=1.14,<1.16`  
  - Python 3.11–3.13: `>=1.15`  

- *pandas* `>=2.2.2`  
- *scikit-learn* `>=1.6`  
- *statsmodels* `>=0.14.2`  
- *lifelines* `>=0.27`  
- *tqdm* `>=4.66`  
- *psutil* `>=5.9`  
- *seaborn* `>=0.13`  
- *formulaic* `>=1.0`  

All dependencies are installed automatically when you install *PyDTS*. 


## Quick Start

The following example demonstrates how to generate synthetic data and fit a `TwoStagesFitter` model.

Detailed definitions and explanations are available in the [methods section](https://tomer1812.github.io/pydts/methods/) of the documentation. 

The function `generate_quick_start_df` simulates a dataset with the following defaults:  

- **Sample size**: `n_patients=10000`  
- **Covariates**: `n_cov=5` independent covariates, each drawn from Uniform(0,1) distribution 
- **Competing events**: `j_events=2` event types  
- **Time scale**: `d_times=14` discrete time intervals  
- **Hazard coefficients** (default values):  
  - $\alpha_{1t}$ = −1 − 0.3 * log(t)  
  - $\alpha_{2t}$ = −1.75 − 0.15 * log(t)
  - $\beta_1$ = −log([0.8, 3, 3, 2.5, 2])  
  - $\beta_2$ = −log([1, 3, 4, 3, 2])  

For each patient, a censoring time $C$ is drawn from Uniform{1, ..., 14}.
The observed time is defined as $X = min(T, C)$, where $T$ is the event time which is sampled based on the covariates of each patient and the hazard coefficients.
If censoring occurs before the event ($C < T$), the event type is set to $J = 0$.

Once the dataset is generated, you can fit a `TwoStagesFitter` to the data (without columns $C$ and $T$ which are not observed in practice).

You can generate synthetic data and fit your first `TwoStagesFitter` model with the following code: 

```python
from pydts.fitters import TwoStagesFitter
from pydts.examples_utils.generate_simulations_data import generate_quick_start_df

# Generate a synthetic dataset with 10,000 patients,
# 5 covariates, 14 discrete time intervals, and 2 competing events
patients_df = generate_quick_start_df(n_patients=10000, n_cov=5, d_times=14, j_events=2, pid_col='pid', seed=0)

# Initialize and fit the discrete-time competing-risk model
fitter = TwoStagesFitter()
fitter.fit(df=patients_df.drop(['C', 'T'], axis=1))

# Display model summary
fitter.print_summary()
```

## Citations
If you found *PyDTS* software useful to your research, please cite the papers:

```bibtex
@article{Meir_PyDTS_2022,
    author = {Meir, Tomer and Gutman, Rom, and Gorfine, Malka},
    doi = {10.48550/arXiv.2204.05731},
    title = {{PyDTS: A Python Package for Discrete Time Survival Analysis with Competing Risks}},
    url = {https://arxiv.org/abs/2204.05731},
    year = {2022}
}

@article{Meir_Gorfine_DTSP_2025,
    author = {Meir, Tomer and Gorfine, Malka},
    doi = {10.1093/biomtc/ujaf040},
    title = {{Discrete-Time Competing-Risks Regression with or without Penalization}},
    year = {2025},
    journal = {Biometrics},
    volume = {81},
    number = {2},
    url = {https://academic.oup.com/biometrics/article/81/2/ujaf040/8120014},
}
```

and please consider starring the project [on GitHub](https://github.com/tomer1812/pydts)

## How to Contribute
1. Open Github issues to suggest new features or to report bugs\errors.
2. Provide feedback on the documentation.
3. Contact Tomer or Rom if you want to add a usage example to the documentation. 
4. If you want to become a developer (thank you, we appreciate it!) - please contact Tomer or Rom for developers' on-boarding. 

Tomer Meir: tomer1812@gmail.com, Rom Gutman: rom.gutman1@gmail.com


## Running Tests Locally

To run the test suite on your local machine, follow these steps:

1. Clone the repository

```bash
git clone https://github.com/tomer1812/pydts.git
cd pydts
```

2. Install the package in editable mode
```bash
pip install -e .
```

3. Run the test suite with Poetry
```bash
pip install poetry
poetry run pytest
```
