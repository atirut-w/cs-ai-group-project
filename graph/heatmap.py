from seaborn import heatmap
from pandas import read_csv
import matplotlib.pyplot as plt

data = read_csv("datasets/data.csv")

heatmap(data.corr(), annot=True)
plt.show()