import math

class Helper:
    @staticmethod
    def convert_to_100_percent(n: int) -> int:
        return math.trunc(n * 100)
    
    @staticmethod
    def get_data_path() -> str:
        return "datasets/data.csv"