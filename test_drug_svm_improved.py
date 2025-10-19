import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from geg.geg import GeneralizedExponentiatedGradient
from geg.constraints import GeneralDemographicParity1
from utils import get_values
import warnings
warnings.filterwarnings('ignore')

# Load drug data
data = pd.read_csv('data/drug.csv')
label, pos_label, priv_group, unpriv_group = get_values('drug')
X = data.drop(columns=[label])
y = data[label]

print(f'Dataset: drug')
print(f'Positive label: {pos_label}')
print(f'Target distribution: {np.bincount(y)}')

# Simple train-test split for quick test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Test baseline SVM with balanced class weights
print('\n=== BASELINE SVM TEST (with balanced class weights) ===')
svm_baseline = SVC(class_weight='balanced')
svm_baseline.fit(X_train, y_train)
y_pred_baseline = svm_baseline.predict(X_test)
print(f'Baseline SVM predictions distribution: {np.bincount(y_pred_baseline)}')
print(f'Baseline SVM classes predicted: {sorted(np.unique(y_pred_baseline))}')

from sklearn.metrics import accuracy_score
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f'Baseline SVM accuracy: {baseline_accuracy:.4f}')

# Test GEG with improved SVM
print('\n=== GEG SVM TEST (with balanced class weights) ===')
try:
    constraint = GeneralDemographicParity1(y_p=pos_label, difference_bound=0.01)
    geg = GeneralizedExponentiatedGradient(
        estimator=SVC(class_weight='balanced', probability=True),
        constraints=constraint,
        eps=1e-3,
        max_iter=10,
        positive_label=pos_label
    )

    print('Training GEG with balanced SVM...')
    geg.fit(X_train.values, y_train.values, sensitive_features=X_train[list(priv_group.keys())].values)

    print('Making predictions...')
    y_pred_geg = geg.predict(X_test.values)
    
    print(f'GEG predictions distribution: {np.bincount(y_pred_geg)}')
    print(f'GEG classes predicted: {sorted(np.unique(y_pred_geg))}')
    
    geg_accuracy = accuracy_score(y_test, y_pred_geg)
    print(f'GEG accuracy: {geg_accuracy:.4f}')
    
    # Check if positive label is now predicted
    if pos_label in np.unique(y_pred_geg):
        print(f'\n✅ Success! Positive label {pos_label} is now being predicted.')
    else:
        print(f'\n⚠️  Positive label {pos_label} is still not predicted, but no error occurred.')
    
except Exception as e:
    print(f'Error occurred: {e}')
    import traceback
    traceback.print_exc()