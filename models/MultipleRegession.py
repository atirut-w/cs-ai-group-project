from module import Module
from typing import override
from pandas import read_csv
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from utils.Helper import Helper


class MultipleRegessionModel(Module):
    def __init__(self):
        super().__init__(
            """
            ตัว model นี้ใช้เป็น multiple linear 
            เป็น model ที่ใช้สำหรับการทำนายราคาของบ้าน 
            โดยให้ x มีหลายคุณสมบัติ คือ area, bedrooms, stories, parking, mainroad,
            guestroom, hotwaterheating, airconditioning, prefarea
            และ y ที่เป็น class คือ price
        """
        )

    @override
    def prepare_dataset(self):
        self.df = read_csv(self.dataset_path)

        # ดึง columns คุณสมบัติของข้อมูล x ที่กำหนดตามเลข index
        # self.selected_cols = [*self.cols[1:3], *self.cols[4:7], *self.cols[8:12]]
        self.selected_cols = [*self.cols[1:]]
        self.x = array(self.df[self.selected_cols].values)
        self.y = array(self.df["price"].values)

    @override
    def train_model(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=50
        )

        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    @override
    def prediction(self) -> None:
        self.y_pred = self.model.predict(self.x_test)

    @override
    def evaluate_model(self) -> None:
        self.set_accuracy_value(r2_score(self.y_test, self.y_pred))
        self.set_error_value(mean_squared_error(self.y_test, self.y_pred))

        print(f"R-squared = {Helper.convert_to_100_percent(self.get_accuracy_value())}%")
        print(f"Mean Squared Error = {self.get_error_value()}")
