from module import Module
from typing import override, List, Optional, Union
from pandas import read_csv
from numpy import array, ndarray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.sparse._matrix import spmatrix
from utils.Helper import Helper


class PolynomialRegressionModel(Module):
    poly: Optional[PolynomialFeatures] = None
    poly_linmodel: Optional[LinearRegression] = None
    x_train_poly: Optional[ndarray] = None
    x_test_poly: Optional[Union[ndarray, spmatrix]] = None

    def __init__(self) -> None:
        super().__init__(
            """
            ตัว model นี้ใช้เป็น polynomial 
            เป็น model ที่ใช้สำหรับการทำนายราคาของบ้าน 
            โดยให้ x มีหลายคุณสมบัติ คือ area, bedrooms, bathrooms, stories
            และ y ที่เป็น class คือ price
        """
        )

    @override
    def prepare_dataset(self) -> None:
        self.df = read_csv(self.dataset_path)

        self.selected_cols = [*self.cols[1:5]]
        self.x = array(self.df[self.selected_cols].values)
        self.y = array(self.df["price"])

    @override
    def train_model(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2
        )

        self.poly = PolynomialFeatures(degree=3)
        self.x_train_poly = self.poly.fit_transform(self.x_train)
        self.x_test_poly = self.poly.transform(self.x_test)

        self.poly.fit(self.x_train_poly, self.y_train)

        self.poly_linmodel = LinearRegression()
        self.model = self.poly_linmodel.fit(self.x_train_poly, self.y_train)

    @override
    def prediction(self) -> None:
        self.y_pred = self.model.predict(self.x_test_poly)

    @override
    def evaluate_model(self) -> None:
        print(
            f"R^2 = {Helper.convert_to_100_percent(r2_score(self.y_test, self.y_pred))}%"
        )
        print(f"Mean Square Error = {mean_squared_error(self.y_test, self.y_pred)}")
        print(f"Mean Absolute Error = {mean_absolute_error(self.y_test, self.y_pred)}")
