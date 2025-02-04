from utils.typing.type import Float
import math

class Helper:
    @staticmethod
    def convert_to_100_percent(n: Float) -> int:
        return math.trunc(n * 100)
    
    @staticmethod
    def get_data_path() -> str:
        return "datasets/data.csv"