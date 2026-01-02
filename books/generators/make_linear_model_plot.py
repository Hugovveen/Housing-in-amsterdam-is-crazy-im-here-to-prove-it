import pandas as pd
import plotly.express as px

df = pd.read_csv("../outputs/linear_model_predictions.csv")

fig = px.scatter(
    df,
    x="ActualPrice",
    y="PredictedPrice",
    title="Linear Regression: Actual vs Predicted Housing Prices",
    opacity=0.6
)

fig.add_shape(
    type="line",
    x0=df["ActualPrice"].min(),
    y0=df["ActualPrice"].min(),
    x1=df["ActualPrice"].max(),
    y1=df["ActualPrice"].max(),
    line=dict(dash="dash")
)

fig.write_html("../outputs/linear_model_actual_vs_predicted.html")
