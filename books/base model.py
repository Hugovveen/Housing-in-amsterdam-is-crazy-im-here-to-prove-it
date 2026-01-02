import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../data/HousingPrices-Amsterdam-August-2021.csv')

df = df.dropna(subset=["Price", "Area", "Room", "Lat", "Lon"]) #dropping missing prices

#################################################################

features = ["Area", "Room", "Lat", "Lon"]

X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

##################################################################

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

results_df = pd.DataFrame({
    "ActualPrice": y_test.values,
    "PredictedPrice": y_pred
})

results_df.to_csv("../outputs/linear_model_predictions.csv", index=False)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test,y_pred)

if __name__ == "__main__":

    print(f"RMSE: {rmse:,.0f}")
    print(f"R²: {r2:.3f}")