from module import Module
from typing import Dict
import sys
from models.LinearRegression import LinearRegressionModel
from models.MultipleRegession import MultipleRegessionModel
from models.PolynomialRegression import PolynomialRegressionModel

modules: Dict[str, Module] = {
    # Add modules here
    'linear_regression': LinearRegressionModel(),
    'multiple_linear_regression': MultipleRegessionModel(),
    'polynomial_regression': PolynomialRegressionModel()
}

if len(sys.argv) == 1:
    print("Usage: python main.py <module_name>")
    print("Available modules:")
    for name, module in modules.items():
        print(f"\t{name}: {module.description}")

# รันคำสั่ง py main.py <ชื่อ model> เช่น py main.py linear_regression
# อ่านค่า argument ที่ส่งไปในคำสั่ง
module_name = sys.argv[1]

# ใช้แฟลก -a (all) รันทุก models
if module_name == "-a": 
    for key in modules:
        print(f'- {key} model')
        module = modules[key]
        module.prepare_dataset()
        module.train_model()
        module.evaluate_model()
        print()
# ถ้ามีชื่อ model ที่ตรงกับ key ใน module ให้รัน model นั้น
elif module_name in modules:
    module = modules[module_name]
    module.prepare_dataset()
    module.train_model()
    module.evaluate_model()
# ถ้าไม่เจอชื่อ model ให้ออกจากโปรแกรม
else:
    print(f"Module {module_name} not found")
    sys.exit(1)