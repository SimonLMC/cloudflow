# cloudflow 

__If you don't know yet what the Predictive Power Score is, please read the following blog post:__

Cloudflow is a Python package that combines two powerful tools for machine learning model management: Cloudpickle and MLflow. 

Cloudpickle is a serialization library that allows Python objects to be serialized and deserialized across different Python processes or even different Python versions. This is especially useful for machine learning models, which can be complex and difficult to share between different environments.

MLflow, on the other hand, is an open source platform for the complete machine learning lifecycle that helps to manage experiments, package code into reproducible runs, and share and deploy models. 

By integrating Cloudpickle with MLflow, Cloudflow provides an easy and efficient way to serialize and save complex machine learning models, track your experiments and their results, and share your models with others in a reproducible and scalable way.

Whether you are working on a single project or collaborating with a team, Cloudflow can help you manage your machine learning models more effectively and streamline your machine learning workflow.

One of the key benefits of Cloudflow is that it makes it easy to save and manage complex combinations models. 
With traditional approaches, it can be challenging to combine multiple models into a single object that can be easily serialized and saved. However, with Cloudpickle-MLflow, you can easily combine multiple models, along with their associated pre-processing and post-processing steps, into a single model object that can be easily serialized and saved. This makes it simple to manage complex combinations of models, which can be very useful when you are working on more advanced machine learning problems.

- [Installation](#installation)
- [Getting started](#getting-started)
- [API](#api)
- [About](#about)

## Installation

> You need Python 3.6 or above.
From the terminal (or Anaconda prompt in Windows), enter:

```bash
pip install cloudflow
```

## Getting started

First, let's create some models:

```python
import pandas as pd

```

    .
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    └── ...



Now, let's save out project:

```python
import pandas as pd

```



## API

### ppscore.score(df, x, y, sample=5_000, cross_validation=4, random_seed=123, invalid_score=0, catch_errors=True)

Calculate the Predictive Power Score (PPS) for "x predicts y"

- The score always ranges from 0 to 1 and is data-type agnostic.

- A score of 0 means that the column x cannot predict the column y better than a naive baseline model.

- A score of 1 means that the column x can perfectly predict the column y given the model.

- A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the baseline model.


#### Parameters

- __df__ : pandas.DataFrame
    - Dataframe that contains the columns x and y
- __x__ : str
    - Name of the column x which acts as the feature
- __y__ : str
    - Name of the column y which acts as the target
- __sample__ : int or `None`
    - Number of rows for sampling. The sampling decreases the calculation time of the PPS.
    If `None` there will be no sampling.
- __cross_validation__ : int
    - Number of iterations during cross-validation. This has the following implications:
    For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same observation. If the limit is increased, the required minimum observations also increase. This is important, because this is the limit when sklearn will throw an error and the PPS cannot be calculated
- __random_seed__ : int or `None`
    - Random seed for the parts of the calculation that require random numbers, e.g. shuffling or sampling.
    If the value is set, the results will be reproducible. If the value is `None` a new random number is drawn at the start of each calculation.
- __invalid_score__ : any
    - The score that is returned when a calculation is not valid, e.g. because the data type was not supported.
- __catch_errors__ : bool
    - If `True` all errors will be catched and reported as `unknown_error` which ensures convenience. If `False` errors will be raised. This is helpful for inspecting and debugging errors.


#### Returns

- __Dict__:
    - A dict that contains multiple fields about the resulting PPS.
    The dict enables introspection into the calculations that have been performed under the hood


