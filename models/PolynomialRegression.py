from module import Module
from typing import override
from pandas import read_csv
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class PolynomialRegressionModel(Module):
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
    def prepare_dataset(self,df) -> None:
        self.df = df
        cols = ['area','bedrooms','bathrooms','stories']
        X = self.df[cols].values
        print(X.shape)
        self.x = array(X)
        self.y = array(self.df['price'])
        # print(self.x)
        # print(self.y)

    @override
    def train_model(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2
        )

        # self.model = LinearRegression()
        # self.model.fit(self.x_train, self.y_train)
        poly = PolynomialFeatures(degree=3)
        x_train_poly = poly.fit_transform(self.x_train)
        self.x_test_poly = poly.transform(self.x_test)
        poly.fit(x_train_poly, self.y_train)
        
        self.poly_linmodel = LinearRegression()
        self.model2 = self.poly_linmodel.fit(x_train_poly, self.y_train)

    @override
    def evaluate_model(self) -> None:
        self.y_predic = self.model2.predict(self.x_test_poly)
        print(f"R^2 = {r2_score(self.y_test, self.y_predic) * 100}")
        print(f"Mean Square Error = {mean_squared_error(self.y_test, self.y_predic)}")
        print(
            f"Mean Absolute Error = {mean_absolute_error(self.y_test, self.y_predic)}"
        )


# model = LinearRegressionModel()
# model.prepare_dataset()
# model.train_model()
# model.evaluate_model()
