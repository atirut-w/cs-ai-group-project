from typing import override
from pandas import read_csv
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.Helper import Helper
from module import Module
from utils.Helper import Helper
from module import Module


class LinearRegressionModel(Module):
    x = None
    y = None
    df = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    model = None
    y_predicted = None
    y_predicted = None

    def __init__(self) -> None:
        super().__init__()
        self.description = """
            ตัว model นี้ใช้เป็น linear 
            เป็น model ที่ใช้ในการทำนายราคาบ้าน 
            โดยให้ x เป็นคุณสมบัติของข้อมูล คือ พื้นที่ของบ้าน (Area)
            และ y ที่เป็น label คือ ราคาของบ้าน (Price)
        """

    @override
    def prepare_dataset(self) -> None:
        self.df = read_csv("datasets/Housing.csv")
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
        self.y_predicted = self.model.predict(self.x_test)

        score = Helper.convert_to_100_percent(r2_score(self.y_test, self.y_predicted))
        print(f"R^2 = {score}%")
        print(f"Mean Square Error = {mean_squared_error(self.y_test, self.y_predicted)}")
        self.y_predicted = self.model.predict(self.x_test)