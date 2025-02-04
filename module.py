from typing import Optional, Union, Any, Tuple, List
from abc import abstractmethod, ABC
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from utils.Helper import Helper


# class Module เป็น abstract class ทุก models ที่สร้างขึ้นมาจะต้องสืบทอด super class ตัวนี้นำไปใช้งาน
class Module(ABC):
    # ประกาศ attributes ทุกตัว
    description: Optional[str] = None
    model: Optional[Union[LinearRegression, DecisionTreeRegressor]] = None
    df: Optional[DataFrame] = None
    x: Optional[NDArray] = None
    y: Optional[NDArray] = None
    x_train: Optional[Any] = None
    x_test: Optional[Any] = None
    y_train: Optional[Any] = None
    y_test: Optional[Any] = None
    y_pred: Optional[ndarray] = None
    dataset_path: Optional[str] = None
    selected_cols: Optional[List[str]] = None
    cols: Tuple[str, ...] = (
        "price",
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus",
    )
    LABEL: str = "price"

    # constructor ของ superclass เมื่อ subclass สืบทอด superclass แล้วต้องส่งค่า arg เข้ามาด้วย
    def __init__(
        self, description: Optional[str] = None, dataset_path: Optional[str] = None
    ):
        self.description = description

        # ถ้าไม่มีการส่งค่า arg ที่เป็น path ของ dataset มาให้ใช้ path เริ่มต้น
        if dataset_path == None:
            self.dataset_path = Helper.get_data_path()

    # abstract methods ของ superclass เมื่อ subclass สืบทอดไปแล้วต้องทำการ implement method เหล่านี้ทั้งหมด
    @abstractmethod
    def prepare_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def train_model(self):
        raise NotImplementedError

    @abstractmethod
    def prediction(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self):
        raise NotImplementedError
