from module import Module
from typing import Dict
import sys
from models.LinearRegression import LinearRegressionModel
from models.MultipleRegession import MultipleRegessionModel
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
    # 'polynomial_regression': None
}

if len(sys.argv) == 1:
    print("Usage: python main.py <module_name>")
    print("Available modules:")
    for name, module in modules.items():
        print(f"\t{name}: {module.description}")
    sys.exit()

# test
for module_name in modules:
    print(module_name)
    module = modules[module_name]
    module.prepare_dataset()
    module.train_model()
    module.evaluate_model()
    