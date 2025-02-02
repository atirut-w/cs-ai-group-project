from pandas import read_csv, DataFrame, to_numeric
from typing import Dict, List
from math import trunc

class DataCleaner:
    df: DataFrame | None = None
    data_path: str = "datasets/Housing.csv"
    result_path: str = "datasets/data.csv"
    is_error: bool = False
    dict_of_yes_no: Dict[str, int] = {"yes": 1, "no": 0}
    dict_of_furnishingstatus: Dict[str, int] = {
        "unfurnished": 0,
        "furnished": 1,
        "semi-furnished": 2,
    }

    def __init__(self) -> None:
        # อ่านค่า csv แล้วเก็บค่าลง attribute dataframe
        self.df = read_csv(self.data_path)
        # print(self.df.describe())
        # print(self.df.head())

        # เก็บรายชื่อ columns ที่ค่าไม่ใช้ตัวเลข
        column_names: List[str] = []
        for column_name in self.df.columns:
            # เช็ค data type ของคอลัมน์ นั้นว่ามีค่าเป็น object หรือเปล่า
            if self.df[column_name].dtype == "object":
                # เพิ่มเข้าใน list
                column_names.append(column_name)

        # แปลงค่าคอลัมน์ที่มีค่าเป็น object(string) แปลงไปเป็นตัวเลข 1 หรือ 0 เพื่อนำ df ไปใช้งาน
        for column_name in column_names:
            # คอลัมน์ furnishingstatus ไม่ได้มีค่าเป็น yes, no ให้ใช้ dict อีกอัน
            if column_name == "furnishingstatus":
                self.convert_str_to_num(
                    column=column_name, dict=self.dict_of_furnishingstatus
                )
            else:
                self.convert_str_to_num(column=column_name, dict=self.dict_of_yes_no)
        # เช็คดูว่าค่าที่เป็น object แปลงเป็น int หมดแล้ว
        # print(self.df.dtypes)

    # method สำหรับคืนค่า error ถ้า True แปลว่ามีค่า error ถ้า False แปลว่าไม่มีค่า error
    def get_error(self) -> bool:
        return self.is_error

    # method สำหรับแก้ไขค่า error ส่งค่า argument เป็น boolean
    def set_error(self, val: bool) -> None:
        self.is_error = val

    # method สำหรับคืนค่า dataframe ที่ทำงานอยู่ใน class นี้
    def get_df(self) -> DataFrame:
        return self.df

    # method สำหรับแปลงค่า string เป็นเลข
    def convert_str_to_num(self, column: str, dict: Dict[str, int]):
        # loop ทุกค่าในแต่ละ cell เปลี่ยนค่า yes = 1 และ no = 0
        for i in self.df.index:
            # นำค่าของ cell มากำหนดเป็น key ให้ dict เพื่อแปลงค่า
            key: str = self.df.loc[i, column]
            # เปลี่ยนค่าใหม่
            self.df.loc[i, column] = dict[key]
            # print(f"{key} = {self.df[column][i]}")

        # แปลงชนิดข้อมูลจาก object เป็น int
        self.df[column] = to_numeric(self.df[column])

    # method สำหรับสร้างไฟล์ csv หลังจากทำ cleaning เสร็จเรียบร้อยแล้ว
    def export_to_csv(self) -> None:
        self.df.to_csv(self.result_path, index=False)
        print("สร้างไฟล์ data.csv สำเร็จ")

    # method เช็คค่าว่างของทุก cell ใน dataframe
    def check_empty_cell(self) -> None:
        # เก็บค่า booleans หากมีค่าเป็น true หมดแปลว่าใน dataframe ไม่มีค่าว่าง หากมีค่า false สักอันแปลว่า
        # มีค่าว่าง
        bools: List[bool] = []
        # วน loop รับค่าชื่อ columns
        for column_name in self.df.columns:
            # เช็คแต่ละหัวคอลัมน์ว่ามีค่าว่างหรือเปล่า ถ้ามีค่าจะมากกว่าหรือเท่ากับ 1
            # หรือ ถ้าไม่มีค่าจะเป็น 0
            if self.df[column_name].isnull().sum() >= 1:
                # มีค่าว่างกำหนดให้ค่า error เป็นจริง
                self.set_error(True)

            # ถ้ามีค่า error จริง ให้ทำ statements นี้
            if self.get_error():
                print(f"คอลัมน์ {column_name} มีค่าว่างทำการแก้ไข")
                # สร้างค่าเฉลี่ยของแต่ล่ะคอลัมน์แต่ตัดเลขทศนิยมออก หากมี cell ที่เป็นค่าว่างให้เอาค่าเฉลี่ยเข้าไปแทนที่ cell นั้น
                mean = trunc(self.df[column_name].mean())
                self.df[column_name].fillna(mean, inplace=True)
                bools.append(False)
                # ขึ้นรอบใหม่ให้ค่า error เป็น False
                self.set_error(False)
            else:
                bools.append(True)

        # หากมีค่า True เท่ากับจำนวนคอลัมน์แปลว่า ไม่มีข้อมูล cell ไหนที่เป็นค่าว่าง
        if bools.count(True) == len(self.df.columns):
            print("ข้อมูลใน dataframe ไม่มีค่าว่าง")
        else:
            print("ข้อมูลใน dataframe มีค่าว่าง แต่ทำการปรับปรุงแก้ไขข้อมูลที่เป็นค่าว่างแล้ว")

    def check_wrong_format(self) -> None: ...

    def check_wrong_data(self) -> None: ...

    def check_duplicate_row(self) -> None:
        ...
        
cleaner = DataCleaner()
cleaner.check_empty_cell()
# cleaner.export_to_csv()
