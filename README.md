[![pypi version](https://img.shields.io/pypi/v/pydts)](https://pypi.org/project/pydts/)
[![Tests](https://github.com/tomer1812/pydts/workflows/Tests/badge.svg)](https://github.com/tomer1812/pydts/actions?workflow=Tests)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://tomer1812.github.io/pydts)
[![codecov](https://codecov.io/gh/tomer1812/pydts/branch/main/graph/badge.svg)](https://codecov.io/gh/tomer1812/pydts)
# Discrete Time Survival Analysis  
A Python package for discrete time survival data analysis with competing risks.

![PyDTS](docs/icon.png)  

[Tomer Meir](https://tomer1812.github.io/), [Rom Gutman](https://github.com/RomGutman), [Malka Gorfine](https://www.tau.ac.il/~gorfinem/) 2022

[Documentation](https://tomer1812.github.io/pydts/)  

## Installation
```console
pip install pydts
```

## Quick Start

```python
from pydts.fitters import TwoStagesFitter
from pydts.examples_utils.generate_simulations_data import generate_quick_start_df
from sklearn.model_selection import train_test_split

patients_df = generate_quick_start_df(n_patients=10000, n_cov=5, d_times=14, j_events=2, pid_col='pid', seed=0)
train_df, test_df = train_test_split(patients_df, test_size=0.25)

fitter = TwoStagesFitter()
fitter.fit(df=train_df.drop(['C', 'T'], axis=1))
fitter.print_summary()

```

## Other Examples
1. [Usage Example](https://tomer1812.github.io/pydts/UsageExample-Intro/)
2. [Hospital Length of Stay Simulation Example](https://tomer1812.github.io/pydts/SimulatedDataset/)
