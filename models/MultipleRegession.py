
from module import Module
from typing import override
from pandas import read_csv
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



class MultipleRegessionModel(Module):
    x = None
    y = None
    df = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    model = None
    y_predic = None
    
    def __init__(self):
        super().__init__()
        
        
        self.description = """
            ตัว model นี้ใช้เป็น multiple linear 
            เป็น model ที่ใช้ในการทำนายเงินเดือนค่าจ้างของพนักงาน 
            โดยให้ x เป็นคุณสมบัติของข้อมูล คือ จำนวนประสบการณ์การทำงาน (Experience_Years)
            และ y ที่เป็น label คือ เงินเดือน (Salary)
        """
    @override
    def prepare_dataset(self,df):
        self.df = df
        cols = ['area','bedrooms','bathrooms','stories','parking','mainroad'
                    ,'guestroom','hotwaterheating','airconditioning'
                    ,'prefarea']
        X = self.df[cols].values
        y = self.df['price'].values
        
        self.X = array(X)
        self.y = array(y)
        
    
    @override
    def train_model(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2
        )

        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    @override
    def evaluate_model(self) -> None:
        self.y_predic = self.model.predict(self.x_test)
        print(f"R^2 = {r2_score(self.y_test, self.y_predic) * 100}")
        print(f"Mean Square Error = {mean_squared_error(self.y_test, self.y_predic)}")
        print(
            f"Mean Absolute Error = {mean_absolute_error(self.y_test, self.y_predic)}"
        )

