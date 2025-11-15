# Generalized Exponentiated Gradient

This repository contains the implementation of the _Generalized Exponentiated Gradient (GEG)_ algorithm, an in-processing bias mitigation method to enhance fairness in binary and multiclass classification tasks.

## Repository Structure

- `geg/`: Contains the main implementation of the GEG algorithm.
- `experiments/`: Contains the replication packages for the experiments reported in the associated research paper.

## Installation

Install Python 3.9 or higher. It is recommended to use a virtual environment like `conda`.

Next, install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

Here is a basic example of how to use the GEG algorithm:

```python
from geg.geg import GeneralizedExponentiatedGradient
from geg.constraints import GeneralDemographicParity1, GeneralEqualizedOdds1, CombinedParityGeneral1
from sklearn.linear_model import LogisticRegression

# Load your dataset
X, y, sensitive_features = load_your_data()
pos_label = 1


# Initialize the base classifier
base_classifier = LogisticRegression(solver='liblinear')

# Initialize the GEG algorithm with a specific fairness constraint
constraint = GeneralDemographicParity1(y_p=pos_label)
geg = GeneralizedExponentiatedGradient(
            estimator=LogisticRegression(),
            constraints=constraint,
            positive_label=pos_label)

# Fit the GEG model
geg.fit(X, y, sensitive_features=sensitive_features)

# Make predictions
predictions = geg.predict(X)
```
