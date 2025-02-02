from module import Module
from typing import Dict
import sys
from models.LinearRegression import LinearRegressionModel
from models.MultipleRegession import MultipleRegessionModel
from models.PolynomialRegression import PolynomialRegressionModel

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


modules: Dict[str, Module] = {
    # Add modules here
    "linear_regression": LinearRegressionModel(),
    'multiple_linear_regression': MultipleRegessionModel(),
    'polynomial_regression': PolynomialRegressionModel()
}

if len(sys.argv) == 1:
    print("Usage: python main.py <module_name>")
    print("Available modules:")
    for name, module in modules.items():
        print(f"\t{name}: {module.description}")

module_name = sys.argv[1]
if module_name not in modules:
    print(f"Module {module_name} not found")
    sys.exit(1)

#load dataset
df = pd.read_csv('datasets/Housing.csv')
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['perfarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 3, 'semi-furnished': 2, 'unfurnished': 1})

scaler = StandardScaler()
cols_to_norm = ['price','area', 'bedrooms', 'bathrooms', 'stories', 'parking','mainroad','guestroom','basement','hotwaterheating','airconditioning','perfarea','furnishingstatus']
scaled_data = scaler.fit(df[cols_to_norm])
df[cols_to_norm] = scaled_data.transform(df[cols_to_norm])


module = modules[module_name]
module.prepare_dataset(df)
module.train_model()
module.evaluate_model()
