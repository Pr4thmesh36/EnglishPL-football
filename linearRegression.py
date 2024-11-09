
# Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data = pd.read_csv("Book5.csv")
independent_vars = data[['Age', '90s', 'Performance']]
dependent_var = data['Market Value']
X_train, X_test, y_train, y_test = train_test_split(independent_vars, dependent_var, test_size=0.25, random_state=26575)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions) # MAE
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
coefficients = model.coef_
intercept = model.intercept_
print('Coefficients:', coefficients)
print('Intercept:', intercept)
all_predictions = model.predict(scaler.transform(independent_vars))
result_df = pd.DataFrame({'Actual': dependent_var, 'Predicted': all_predictions})
print(result_df)
plt.scatter(result_df['Actual'], result_df['Predicted'], label='Predicted Values')
plt.plot([min(dependent_var), max(dependent_var)], [min(dependent_var), max(dependent_var)], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.legend()
plt.show()
# Individual player value prediction for Linear Regression
new_data_lr = pd.DataFrame({'Age': [22], '90s': [30.8], 'Performance': [7.76]})
input_data_scaled_lr = scaler.transform(new_data_lr)
predicted_market_value_lr = model.predict(input_data_scaled_lr)
print('Predicted Market Value (Linear Regression) for New Data:', predicted_market_value_lr[0])
