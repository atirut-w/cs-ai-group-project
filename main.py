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

# #load dataset
df = pd.read_csv('datasets/data.csv')


module = modules[module_name]
module.prepare_dataset(df)
module.train_model()
module.evaluate_model()
