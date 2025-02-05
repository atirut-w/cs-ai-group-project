import matplotlib.pyplot as plt
from typing import Optional, Union, List
from numpy import ndarray
from utils.typing.type import Float

# class GraphMaker สำหรับการวาดกราฟเส้น
class GraphMaker:
    # กำหนดค่า attributes ทั้งหมด
    # ใช้ค่า r-square
    accuracy_values: Optional[Union[List[Float], List[ndarray]]] = None
    # ใช้ค่า error เป็น mean square error
    error_values: Optional[Union[List[Float], List[ndarray]]] = None
    xlabel: str = "Regression Models"
    ylabel: str = "R Square"
    ylabel2: str = "Mean Square Error"
    title: str = "R Square VS Mean Square Error"
    model_names: List[str] = ['Linear', 'Multiple Linear', 'Polynomial', 'Decision Tree']

    # สร้าง constructor โดยมีค่า args ที่ต้องส่งคือ ค่า accuracy และ ค่า error ส่งมาเป็น list
    def __init__(
        self,
        acc_vals: Optional[Union[List[Float], List[ndarray]]] = None,
        err_vals: Optional[Union[List[Float], List[ndarray]]] = None,
    ) -> None:
        self.set_accuracy_values(acc_vals)
        self.set_error_values(err_vals)

    # methods สำหรับการวาดกราฟค่า accuracy (R2 score) ของ model โดยเปรียบเทียบแต่ล่ะ model
    def create_accuracy_graph(self) -> None:        
        plt.plot(self.model_names, self.accuracy_values)
        plt.title("Compare the R Square values o​f each model")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()

    # methods สำหรับการวาดกราฟค่า error (mean square error) ของ model โดยเปรียบเทียบแต่ล่ะ model
    def create_error_graph(self) -> None:        
        plt.plot(self.model_names, self.error_values)
        plt.title("Compare the Mean Square Error values ​​of each model")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel2)
        plt.show()
        
    # method สำหรับส่งค่า r square เป็น list
    def get_accuracy_values(self) -> Optional[Union[List[Float], List[ndarray]]]:
        return self.accuracy_values

    # method สำหรับส่งค่า mean square error เป็น list
    def get_error_values(self) -> Optional[Union[List[Float], List[ndarray]]]:
        return self.error_values

    # method สำหรับเปลี่ยน ค่า r square
    def set_accuracy_values(self, acc_vals: Union[List[Float], List[ndarray]]) -> None:
        if not (acc_vals == None):
            self.accuracy_values = acc_vals

    # method สำหรับเปลี่ยนค่า error 
    def set_error_values(self, err_vals: Union[List[Float], List[ndarray]]) -> None:
        if not (err_vals == None):
            self.error_values = err_vals
