
#XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("Book5.csv")
independent_vars = data[['Age', '90s', 'Performance']]
dependent_var = data['Market Value']
X_train, X_test, y_train, y_test = train_test_split(
 independent_vars, dependent_var, train_size=0.75, random_state=47
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_xgb = XGBRegressor(objective='reg:squarederror', random_state=47)
model_xgb.fit(X_train_scaled, y_train)
predictions_xgb = model_xgb.predict(X_train_scaled)
mse_xgb = mean_squared_error(y_train, predictions_xgb)
r2_xgb = r2_score(y_train, predictions_xgb)
rmse_xgb = mean_squared_error(y_train, predictions_xgb, squared=False)
mae_xgb = mean_absolute_error(y_train, predictions_xgb)
print(f'Root Mean Squared Error: {rmse_xgb}')
print(f'Mean Absolute Error: {mae_xgb}')
print(f'Mean Squared Error: {mse_xgb}')
print(f'R-squared: {r2_xgb}')
all_predictions_xgb = model_xgb.predict(scaler.transform(independent_vars))
result_df_xgb = pd.DataFrame({'Actual': dependent_var, 'Predicted (XGBoost)': all_predictions_xgb})
print('\nTable of Actual vs Predicted Values (XGBoost):')
print(result_df_xgb)
plt.scatter(y_train, predictions_xgb)
plt.xlabel('Actual Market Value')
plt.ylabel('Predicted Market Value (XGBoost)')
plt.title('Actual vs. Predicted Market Values (XGBoost)')
regression_line_xgb = np.polyfit(y_train, predictions_xgb, 1)
plt.plot(y_train, np.polyval(regression_line_xgb, y_train), color='red', linewidth=2, label='Regression Line (XGBoost)')
plt.legend()
plt.show()
new_data_xgb = pd.DataFrame({'Age': [22], '90s': [30.8], 'Performance': [7.76]})
input_data_scaled_xgb = scaler.transform(new_data_xgb)
predicted_market_value_xgb = model_xgb.predict(input_data_scaled_xgb)
print('Predicted Market Value (XGBoost) for New Data:', predicted_market_value_xgb[0])
