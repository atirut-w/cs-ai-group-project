from seaborn import heatmap
from seaborn.colors.crayons import crayons
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt

# อ่านข้อมูลใน dataset
data: DataFrame = read_csv("datasets/data.csv")

# สร้างกราฟความสัมพันธ์ของแต่ละคอลัมน์ใน dataset
heatmap(data.corr(), annot=True, linecolor=crayons["Black"], linewidths=2, cbar=True)
# แสดงกราฟ
plt.show()
