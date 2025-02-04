from module import Module
from typing import override
from pandas import read_csv
from numpy import array
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.Helper import Helper


class DecisionTreeRegressionModel(Module):
    def __init__(self) -> None:
        super().__init__(
            """
            ตัว model นี้ใช้เป็น decision tree
            เป็น model ที่ใช้สำหรับการทำนายราคาของบ้าน 
            โดยให้ x มี 2 คุณสมบัติ คือ area กับ bedrooms
            และ y ที่เป็น class คือ price
        """
        )

    @override
    def prepare_dataset(self) -> None:
        self.df = read_csv(self.dataset_path)

        self.selected_cols = [self.cols[1], self.cols[2]]
        self.x = self.df[self.selected_cols].values
        self.y = self.df["price"].values

        self.x = array(self.x)
        self.y = array(self.y)

    @override
    def train_model(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )
        self.model = DecisionTreeRegressor(random_state=42)
        self.model.fit(self.x_train, self.y_train)

    @override
    def prediction(self) -> None:
        self.y_pred = self.model.predict(self.x_test)

    @override
    def evaluate_model(self) -> None:
        print(
            f"R^2 = {Helper.convert_to_100_percent(r2_score(self.y_test, self.y_pred))}%"
        )
        print(f"Mean Square Error = {mean_squared_error(self.y_test, self.y_pred)}")
        print(f"Mean Absolute Error = {mean_absolute_error(self.y_test, self.y_pred)}")
