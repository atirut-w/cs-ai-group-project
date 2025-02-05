from typing import Tuple, List
from graph.GraphMaker import GraphMaker
from module import Module
from models.LinearRegression import LinearRegressionModel
from models.MultipleRegession import MultipleRegessionModel
from models.PolynomialRegression import PolynomialRegressionModel
from models.DecisionTreeRegressorModel import DecisionTreeRegressionModel
from utils.typing.type import Float

# นำเข้า models ทั้งหมดแต่ละ  module เก็บไว้ใน tuple
models: Tuple[Module, ...] = (
    LinearRegressionModel(),
    MultipleRegessionModel(),
    PolynomialRegressionModel(),
    DecisionTreeRegressionModel()
)

# list สำหรับเก็บค่า accuracy กับค่า error
acc_vals: List[Float] = []
err_vals: List[Float] = []

# วน loop รับค่า element ที่เป็น model ของแต่ล่ะตัว
for model in models:
    # เรียกใช้ method ของ model โดยให้ทำตามขั้นตอน
    model.prepare_dataset()
    model.train_model()
    model.prediction()    
    model.evaluate_model()
    # เก็บค่า accuracy และ error ของแต่ล่ะ model
    acc_vals.append(model.get_accuracy_value())
    err_vals.append(model.get_error_value())
    
# สร้าง instance ของ GraphMaker โดยส่งค่า args ไปด้วย
maker: GraphMaker = GraphMaker(acc_vals, err_vals)
# เรียกใช้ methods สร้าง graph
maker.create_accuracy_graph()
maker.create_error_graph()