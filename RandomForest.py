
# RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("Book5.csv")
independent_vars = data[['Age', '90s', 'Performance']]
dependent_var = data['Market Value']
X_train, X_test, y_train, y_test = train_test_split(
 independent_vars, dependent_var, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_rf = RandomForestRegressor(n_estimators=3740, random_state=11)
model_rf.fit(X_train_scaled, y_train)
predictions_rf = model_rf.predict(X_train_scaled)
mse_rf = mean_squared_error(y_train, predictions_rf)
r2_rf = r2_score(y_train, predictions_rf)
rmse_rf = mean_squared_error(y_train, predictions_rf, squared=False)
mae_rf = mean_absolute_error(y_train, predictions_rf)
print(f'Root Mean Squared Error: {rmse_rf}')
print(f'Mean Absolute Error: {mae_rf}')
print(f'Mean Squared Error: {mse_rf}')
print(f'R-squared: {r2_rf}')
all_predictions_rf = model_rf.predict(scaler.transform(independent_vars))
result_df_rf = pd.DataFrame({'Actual': dependent_var, 'Predicted (Random Forest)': all_predictions_rf})
print('\nTable of Actual vs Predicted Values (Random Forest):')
print(result_df_rf)
plt.scatter(y_train, predictions_rf)
plt.xlabel('Actual Market Value')
plt.ylabel('Predicted Market Value (Random Forest)')
plt.title('Actual vs. Predicted Market Values (Random Forest)')
regression_line_rf = np.polyfit(y_train, predictions_rf, 1)
plt.plot(y_train, np.polyval(regression_line_rf, y_train), color='red', linewidth=2, label='Regression Line (Random Forest)')
plt.legend()
plt.show()
# Individual player value prediction for Random Forest
new_data_rf = pd.DataFrame({'Age': [22], '90s': [30.8], 'Performance': [7.76]})
new_data_scaled_rf = scaler.transform(new_data_rf)
predicted_market_value_rf = model_rf.predict(new_data_scaled_rf)
print('\nPredicted Market Value (Random Forest) for New Data:', predicted_market_value_rf[0])
