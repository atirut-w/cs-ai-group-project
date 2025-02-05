import matplotlib.pyplot as plt
from typing import Optional, Union, List
from numpy import ndarray
from utils.typing.type import Float


class GraphMaker:
    # ใช้ค่า r-square
    accuracy_values: Optional[Union[List[Float], List[ndarray]]] = None
    # ใช้ค่า error เป็น mean square error
    error_values: Optional[Union[List[Float], List[ndarray]]] = None
    xlabel: str = "Regression Models"
    ylabel: str = "R Square"
    ylabel2: str = "Mean Square Error"
    title: str = "R Square VS Mean Square Error"
    model_names: List[str] = ['Linear', 'Multiple Linear', 'Polynomial', 'Decision Tree']

    def __init__(
        self,
        acc_vals: Optional[Union[List[Float], List[ndarray]]] = None,
        err_vals: Optional[Union[List[Float], List[ndarray]]] = None,
    ) -> None:
        self.set_accuracy_values(acc_vals)
        self.set_error_values(err_vals)

    def create_accuracy_graph(self) -> None:        
        plt.plot(self.model_names, self.accuracy_values)
        plt.title("Compare the R Square values o​f each model")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()

    def create_error_graph(self) -> None:        
        plt.plot(self.model_names, self.error_values)
        plt.title("Compare the Mean Square Error values ​​of each model")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel2)
        plt.show()
        
    def get_accuracy_values(self) -> Optional[Union[List[Float], List[ndarray]]]:
        return self.accuracy_values

    def get_error_values(self) -> Optional[Union[List[Float], List[ndarray]]]:
        return self.error_values

    def set_accuracy_values(self, acc_vals: Union[List[Float], List[ndarray]]) -> None:
        if not (acc_vals == None):
            self.accuracy_values = acc_vals

    def set_error_values(self, err_vals: Union[List[Float], List[ndarray]]) -> None:
        if not (err_vals == None):
            self.error_values = err_vals
