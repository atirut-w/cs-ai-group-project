import matplotlib.pyplot as plt
from typing import Optional, Union, List
from numpy import ndarray
from utils.typing.type import Float


class GraphMaker:
    # ใช้ค่า r-square
    accuracy_values: Optional[Union[List[Float], List[ndarray]]] = None
    # ใช้ค่า error เป็น mean square error
    error_values: Optional[Union[List[Float], List[ndarray]]] = None
    xlabel: str = "Accuracy values"
    ylabel: str = "Error values"
    title: str = "Accuracy values VS Error values"

    def __init__(
        self,
        acc_vals: Optional[Union[List[Float], List[ndarray]]] = None,
        err_vals: Optional[Union[List[Float], List[ndarray]]] = None,
    ) -> None:
        self.set_accuracy_values(acc_vals)
        self.set_error_values(err_vals)

    def create_graph(self) -> None:
        plt.plot(self.accuracy_values, self.error_values)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
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
