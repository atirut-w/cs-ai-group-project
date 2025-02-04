from module import Module
from typing import Dict
import sys
from models.LinearRegression import LinearRegressionModel
from models.MultipleRegession import MultipleRegessionModel
from models.PolynomialRegression import PolynomialRegressionModel
from models.DecisionTreeRegressorModel import DecisionTreeRegressionModel
from pathlib import Path
from data_cleaning.DataCleaner import DataCleaner

# ถ้ายังไม่มีข้อมูล dataset ที่ยังไม่ได้ clean ให้ clean ข้อมูลก่อนรันตัว model
dataset_file: Path = Path('datasets/data.csv')
# ถ้าไม่มีไฟล์ data.csv ให้สร้างไฟล์ data ที่ clean ข้อมูลเรียบร้อยแล้ว
if not dataset_file.is_file():
    # สร้าง object ของ data cleaner
    cleaner: DataCleaner = DataCleaner()
    # เรียกใช้ methods จาก object เพื่อทำการ clean datset อันเก่า
    cleaner.check_empty_cell()
    cleaner.check_duplicate_row()
    cleaner.check_wrong_format()
    # สร้างไฟล์ dataset อันใหม่เพื่อนำ dataset นี้ไปใช้ train model
    cleaner.export_to_csv()
    
# สร้าง modules ในการเก็บ regression models ของแต่ละ module
modules: Dict[str, Module] = {
    'linear_regression': LinearRegressionModel(),
    'multiple_linear_regression': MultipleRegessionModel(),
    'polynomial_regression': PolynomialRegressionModel(),
    'decision_tree_regression': DecisionTreeRegressionModel()
}

if len(sys.argv) == 1:
    print("Usage: python main.py <module_name>")
    print("Available modules:")
    for name, module in modules.items():
        print(f"\t{name}: {module.description}")

# เขียน try catch ดักไว้ถ้าไม่ป้อนค่า argument พร้อมกับคำสั่งจะเกิด exception ขึ้น
try:
    # รันคำสั่ง py main.py <ชื่อ model> เช่น py main.py linear_regression
    # อ่านค่า argument ที่ส่งไปในคำสั่ง
    module_name: str = sys.argv[1]
    # ใช้แฟลก -a (all) รันทุก models
    if module_name == "-a": 
        for key in modules:
            print(f'- {key} model')
            module: Module = modules[key]
            module.prepare_dataset()
            module.train_model()
            module.prediction()
            module.evaluate_model()
            print()
    # ถ้ามีชื่อ model ที่ตรงกับ key ใน module ให้รัน model นั้น
    elif module_name in modules:
        module: Module = modules[module_name]
        module.prepare_dataset()
        module.train_model()
        module.prediction()    
        module.evaluate_model()
    # ถ้าไม่เจอชื่อ model ให้ออกจากโปรแกรม
    else:
        raise Exception(f"Module {module_name} not found!");
except IndexError as err:
    print('Arguments must be entered!')
except Exception as err:
    print(err.__str__())