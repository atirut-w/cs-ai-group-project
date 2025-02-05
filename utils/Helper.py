from utils.typing.type import Float
import math

# class Helper จะมี utility methods สำหรับนำไปใช้ในส่วนอื่นๆของ modules
class Helper:
    # methods สำหรับการแปลงเลขทศนิยมเป็นค่าเปอร์เซ็น
    @staticmethod
    def convert_to_100_percent(n: Float) -> int:
        return math.trunc(n * 100)
    
    # methods สำหรับการส่ง path ไฟล์ dataset ของข้อมูล
    @staticmethod
    def get_data_path() -> str:
        return "datasets/data.csv"