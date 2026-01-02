## Amsterdam Housing Prices – One-Day ML Exploration

This project explores housing prices in Amsterdam using linear regression models.

### Data
- Source: Amsterdam housing listings (August 2021)
- Target: Property price
- Features: Area, number of rooms, latitude, longitude

### Approach
1. Exploratory data analysis revealed strong right-skew and heteroscedasticity.
2. A baseline linear regression model was trained on raw prices.
3. A log transformation was applied to the target to stabilize variance.
4. Model diagnostics were evaluated using residual plots.

### Key Findings
- Size and location explain a large portion of price variance.
- The log-transformed model produces more consistent errors across price ranges.
- Linear models struggle with extreme high-priced properties.

### Outputs
- `outputs/log_actual_vs_predicted.html`
- `outputs/log_residuals.html`


Next steps:
- Add Zip as categorical feature
- Compare Ridge vs Lasso on log-model
- Try one tree-based model (Random Forest)