from pandas import read_csv, DataFrame, to_numeric
import pandas as pd
from typing import Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
from math import trunc
from utils.Helper import Helper


# DataCleaner เป็น class ที่ทำหน้าที่ในการ clean ข้อมูลใน dataset ก่อนนำข้อมูลไปใช้งาน
class DataCleaner:
    df: Optional[DataFrame] = None
    scaler: MinMaxScaler = MinMaxScaler()
    is_error: bool = False
    column_names: List[str] = []
    dict_of_yes_no: Dict[str, int] = {"yes": 1, "no": 0}
    dict_of_furnishingstatus: Dict[str, int] = {
        "unfurnished": 0,
        "furnished": 1,
        "semi-furnished": 2,
    }

    def __init__(self) -> None:
        # อ่านค่า csv แล้วเก็บค่าลง attribute dataframe
        self.df = read_csv("datasets/Housing.csv")

        # เก็บรายชื่อ columns ที่ค่าไม่ใช้ตัวเลข
        for column_name in self.df.columns:
            # เช็ค data type ของคอลัมน์ นั้นว่ามีค่าเป็น object หรือเปล่า
            if self.df[column_name].dtype == "object":
                # เพิ่มเข้าใน list
                self.column_names.append(column_name)

        # แปลงค่าคอลัมน์ที่มีค่าเป็น object(string) แปลงไปเป็นตัวเลข 1 หรือ 0 เพื่อนำ df ไปใช้งาน
        for column_name in self.column_names:
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
    def get_df(self) -> Optional[DataFrame]:
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

    # method สำหรับการ normalize ข้อมูลให้ข้อมูลอยู่ในช่วง scale เดียวกัน
    def normalized(self) -> None:
        self.df = DataFrame(
            self.scaler.fit_transform(self.df), columns=self.df.columns, dtype="float16"
        )

    # method สำหรับเช็คค่าว่างของทุก cell ใน dataframe
    def check_empty_cell(self) -> None:
        # เก็บค่า booleans หากมีค่าเป็น true หมดแปลว่าใน dataframe ไม่มีค่าว่าง หากมีค่า false สักอันแปลว่า มีค่าว่าง
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
                mean: int = trunc(self.df[column_name].mean())
                self.df[column_name].fillna(mean, inplace=True)
                bools.append(False)
                # ขึ้นรอบใหม่ให้ค่า error เป็น False
                self.set_error(False)
            else:
                bools.append(True)

        # หากมีค่า True เท่ากับจำนวนคอลัมน์แปลว่า ไม่มีข้อมูล cell ไหนที่เป็นค่าว่าง
        self.set_error(bools.count(True) == len(self.df.columns))

        if self.get_error():
            print("ข้อมูลใน dataframe ไม่มีค่าว่าง")
        else:
            print("ข้อมูลใน dataframe มีค่าว่าง แต่ทำการปรับปรุงแก้ไขข้อมูลที่เป็นค่าว่างแล้ว")

    # method สำหรับแก้ไขรูปแบบของชนิดข้อมูล
    def check_wrong_format(self) -> None:
        if len(self.column_names) >= 1:
            self.df[self.column_names] = self.df[self.column_names].map(
                lambda x: pd.to_numeric(x)
            )
            print("เเปลงค่าเป็นตัวเลขเเล้ว")
        else:
            print("ชนิดของข้อมูลแต่ละคอลัมน์ถูกต้องแล้ว")

    # method สำหรับเช็คแถวว่ามีค่าซ้ำกันหรือไม่
    def check_duplicate_row(self) -> None:
        self.set_error(True in self.df.duplicated())
        # เช็คว่าถ้ามีค่า True ซึ่งเป็นค่าซ้ำในแถว
        if self.get_error():
            self.df.drop_duplicates(inplace=True)
            print("มีค่าซ้ำในแถวทำการลบค่าซ้ำออกจากใน dataframe")
        else:
            print("ไม่มีค่าซ้ำในแถว")

    # method สำหรับสร้างไฟล์ csv หลังจากทำ cleaning เสร็จเรียบร้อยแล้ว
    def export_to_csv(self) -> None:
        self.df.to_csv(Helper.get_data_path(), index=False)
        print("สร้างไฟล์ data.csv สำเร็จ")
