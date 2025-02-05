from seaborn import heatmap
from seaborn.colors.crayons import crayons
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt

data: DataFrame = read_csv("datasets/data.csv")

heatmap(data.corr(), annot=True, linecolor=crayons["Black"], linewidths=2, cbar=True)
plt.show()
