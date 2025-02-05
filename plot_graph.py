from typing import Tuple, List
from graph.GraphMaker import GraphMaker
from module import Module
from models.LinearRegression import LinearRegressionModel
from models.MultipleRegession import MultipleRegessionModel
from models.PolynomialRegression import PolynomialRegressionModel
from models.DecisionTreeRegressorModel import DecisionTreeRegressionModel
from utils.typing.type import Float

models: Tuple[Module, ...] = (
    LinearRegressionModel(),
    MultipleRegessionModel(),
    PolynomialRegressionModel(),
    DecisionTreeRegressionModel()
)

acc_vals: List[Float] = []
err_vals: List[Float] = []

for model in models:
    model.prepare_dataset()
    model.train_model()
    model.prediction()    
    model.evaluate_model()
    acc_vals.append(model.get_accuracy_value())
    err_vals.append(model.get_error_value())
    
maker: GraphMaker = GraphMaker(acc_vals, err_vals)
maker.create_accuracy_graph()
maker.create_error_graph()