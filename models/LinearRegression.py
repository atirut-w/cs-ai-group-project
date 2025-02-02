from module import Module
from typing import override
from pandas import read_csv
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.Helper import Helper


class LinearRegressionModel(Module):
    x = None
    y = None
    df = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    model = None
    y_predic = None

    def __init__(self) -> None:
        super().__init__()
        self.description = """
            ตัว model นี้ใช้เป็น linear 
            เป็น model ที่ใช้ในการทำนายเงินเดือนค่าจ้างของพนักงาน 
            โดยให้ x เป็นคุณสมบัติของข้อมูล คือ จำนวนประสบการณ์การทำงาน (Experience_Years)
            และ y ที่เป็น label คือ เงินเดือน (Salary)
        """

    @override
    def prepare_dataset(self) -> None:
        self.df = read_csv("datasets/data.csv")
        self.x = array(self.df[["area"]])
        self.y = array(self.df[["price"]])

    @override
    def train_model(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2
        )

        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    @override
    def evaluate_model(self) -> None:
        self.y_predic = self.model.predict(self.x_test)
        print(
            f"R^2 = {Helper.convert_to_100_percent(r2_score(self.y_test, self.y_predic))}%"
        )
        print(f"Mean Square Error = {mean_squared_error(self.y_test, self.y_predic)}")
        print(
            f"Mean Absolute Error = {mean_absolute_error(self.y_test, self.y_predic)}"
        )
