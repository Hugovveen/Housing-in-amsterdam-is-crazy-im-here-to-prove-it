import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px

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

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_canblog))
r2_log = r2_score(y_test, y_pred_log)

plot_df = pd.DataFrame({
    "Actual Log-Price": y_test,
    "Predicted Log-Price": y_pred_log
})

fig1 = px.scatter(
    plot_df,
    x="Actual Log-Price",
    y="Predicted Log-Price",
    title="Log-Linear Model: Actual vs Predicted Prices",
    opacity=0.6
)

# Perfect prediction line
min_val = plot_df.min().min()
max_val = plot_df.max().max()

fig1.add_shape(
    type="line",
    x0=min_val,
    y0=min_val,
    x1=max_val,
    y1=max_val,
    line=dict(dash="dash")
)

fig1.write_html("log_linear_actual_vs_predicted.html")
