from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Predict and Evaluate for DT
y_pred = dt_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Decision Tree MSE:", mse)
print("Decision Tree R²:", r2)
print("Decision Tree MAE:", mae)


# Predict and Evaluate for RF
y_pred_rf = rf_pipeline.predict(X_test)

print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))

print(Y.mean())

#Plotting
import matplotlib.pyplot as plt

# Colors
actual_color = 'black'
dt_color = 'blue'
rf_color = 'green'

plt.figure(figsize=(8,6))

# Plot perfect prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')

# Scatter plots
plt.scatter(y_test, y_test, color=actual_color, alpha=0.3, label='Actual Prices')  # actuals along y=x line
plt.scatter(y_test, y_pred, color=dt_color, alpha=0.6, label='Decision Tree')
plt.scatter(y_test, y_pred_rf, color=rf_color, alpha=0.6, label='Random Forest')

# Labels and title
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices: Decision Tree vs Random Forest')
plt.legend()
plt.grid(True)

# Save figure
plt.savefig('actual_dt_rf_pred_vs_actual.png')
plt.close()
