import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../data/HousingPrices-Amsterdam-August-2021.csv')
df = df.dropna(subset=["Price", "Area", "Room", "Lat", "Lon"]) #dropping missing prices

###################################################################

df["LogPrice"] = np.log(df["Price"])

features = ["Area", "Room", "Lat", "Lon"]

X = df[features]
y_log = df["LogPrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

###################################################################

model_log = LinearRegression()
model_log.fit(X_train, y_train)

y_pred_log = model_log.predict(X_test)

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
r2_log = r2_score(y_test, y_pred_log)

print(f"Log-RMSE: {rmse_log:.3f}")
print(f"Log-R²: {r2_log:.3f}")

residuals_log = y_test - y_pred_log

plt.scatter(y_pred_log, residuals_log, alpha=0.6)
plt.axhline(0)
plt.xlabel("Predicted Log-Price")
plt.ylabel("Residual (Log)")
plt.title("Residuals vs Predicted (Log-Price Model)")
plt.show()