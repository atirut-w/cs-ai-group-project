import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv("datasets/Housing.csv")
X = np.array(data[["area"]])
y = np.array(data[["price"]])
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("R-squared value:", r2_score(y_test, y_predict))
print("Mean Squared Error value:", mean_squared_error(y_test, y_predict))
print("Mean Absolute Error value:", mean_absolute_error(y_test, y_predict))
