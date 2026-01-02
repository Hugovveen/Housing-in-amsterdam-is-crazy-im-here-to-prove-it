import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/HousingPrices-Amsterdam-August-2021.csv")

df = df.dropna(subset=["Price", "Area", "Room", "Lat", "Lon"])
df["LogPrice"] = np.log(df["Price"])

features = ["Area", "Room", "Lat", "Lon"]
X = df[features]
y = df["LogPrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plot_df = pd.DataFrame({
    "Actual Log Price": y_test.values,
    "Predicted Log Price": y_pred
})

fig = px.scatter(
    plot_df,
    x="Actual Log Price",
    y="Predicted Log Price",
    title="Log-Linear Model: Actual vs Predicted",
    opacity=0.6
)

min_val = plot_df.min().min()
max_val = plot_df.max().max()

fig.add_shape(
    type="line",
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(dash="dash")
)

fig.write_html("log_actual_vs_predicted.html")
